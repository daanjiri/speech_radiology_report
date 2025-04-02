import shutil
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.sox_effects import apply_effects_tensor
import librosa
import numpy as np
from tqdm import tqdm
import random
import whisper
from pathlib import Path
import json
from openai import OpenAI
import io
from dotenv import load_dotenv

def compute_snr(waveform, n_fft=1024, hop_length=512, noise_frames=5, percentile=10):
    """
    Computes an approximate SNR(signal to noise ratio) (in dB) for the given waveform using a more robust method.
    
    Args:
        waveform (Tensor): Audio tensor of shape [channels, samples]. Should be mono.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        noise_frames (int): Number of initial frames to use as noise estimate.
        percentile (int): Percentile to use for noise estimation (lower = more sensitive)
    
    Returns:
        float: Estimated SNR in decibels.
    """
    waveform = waveform.squeeze()
    
    window = torch.hann_window(n_fft).to(waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                     window=window, return_complex=True)
    
    magnitude = torch.abs(stft)
    mag_np = magnitude.cpu().numpy()
    frame_energies = np.mean(mag_np ** 2, axis=0)
    noise_threshold = np.percentile(frame_energies, percentile)
    noise_frames_mask = frame_energies <= noise_threshold
    
    if np.sum(noise_frames_mask) > 0:
        noise_indices = np.where(noise_frames_mask)[0]
        noise_magnitude = magnitude[:, noise_indices]
        noise_power = torch.mean(noise_magnitude ** 2).item()
    else:
        noise_power = torch.mean(magnitude[:, :noise_frames] ** 2).item()
    
    signal_power = torch.mean(magnitude ** 2).item()
    
    epsilon = 1e-8
    snr = 10 * np.log10(signal_power / (noise_power + epsilon))
    
    return snr

class AudioTransform:
    def __init__(self, 
                 noise_reduction_methods=None,
                 compress_internal=False,
                 silence_threshold_db=30, 
                 max_silence_duration=0.01,
                 n_fft=1024, 
                 hop_length=512, 
                 noise_frames=5,
                 wiener_beta=0.002, 
                 highpass_cutoff=100,
                 device='cpu'):
        """
        Audio transformation class for preprocessing audio data.
        
        Args:
            noise_reduction_methods (list, optional): List containing 'spectral' and/or 'wiener'
                                                    to apply noise reduction techniques in sequence.
            compress_internal (bool): If True, compress long internal silences.
            silence_threshold_db (float): dB threshold for detecting silence for compression.
            max_silence_duration (float): Max allowed internal silence duration (in seconds).
            n_fft (int): FFT window size for spectral operations.
            hop_length (int): Hop length for STFT.
            noise_frames (int): Number of initial frames to use as noise estimate.
            wiener_beta (float): Minimum gain factor for Wiener filter.
            highpass_cutoff (int): Cutoff frequency for high-pass filter.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        if noise_reduction_methods is None:
            self.noise_reduction_methods = ['spectral', 'wiener']
        elif isinstance(noise_reduction_methods, str):
            self.noise_reduction_methods = [noise_reduction_methods]
        else:
            self.noise_reduction_methods = noise_reduction_methods
            
        self.compress_internal = compress_internal
        self.silence_threshold_db = silence_threshold_db
        self.max_silence_duration = max_silence_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_frames = noise_frames
        self.wiener_beta = wiener_beta
        self.highpass_cutoff = highpass_cutoff
        self.device = device
        
    def __call__(self, waveform, sample_rate):
        """
        Apply the audio transformation pipeline.
        
        Args:
            waveform (Tensor): Audio tensor.
            sample_rate (int): Sample rate of the audio.
            
        Returns:
            Tensor: Preprocessed waveform.
        """
        waveform = waveform.to(self.device)
        return self.preprocess_transform(waveform, sample_rate)
    
    def spectral_subtraction_transform(self, waveform, sample_rate):
        """
        Applies spectral subtraction to reduce stationary noise using PyTorch operations.
        Assumes that the first few frames (noise_frames) contain only noise.
        """
        waveform = waveform.squeeze()
        
        window = torch.hann_window(self.n_fft).to(waveform.device)
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=window, return_complex=True)
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        noise_mag = torch.mean(magnitude[:, :self.noise_frames], dim=1, keepdim=True)
        noise_mag = noise_mag * 1.2
        subtracted = magnitude - noise_mag
        subtracted = torch.clamp(subtracted, min=0)
        
        D_clean = subtracted * torch.exp(1j * phase)
        y_clean = torch.istft(D_clean, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        
        return y_clean.unsqueeze(0)

    def wiener_filter_transform(self, waveform, sample_rate):
        """
        Applies a basic Wiener filter to reduce noise using PyTorch operations.
        The gain is computed for each frequency bin based on an estimated SNR.
        """
        waveform = waveform.squeeze()
        
        window = torch.hann_window(self.n_fft).to(waveform.device)
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=window, return_complex=True)
        
        magnitude = torch.abs(stft)
        # phase = torch.angle(stft)
        noise_mag = torch.mean(magnitude[:, :self.noise_frames], dim=1, keepdim=True)
        noise_mag = noise_mag * 1.1
        
        power_spec = magnitude ** 2
        noise_power = noise_mag ** 2
        eps = 1e-8
        
        gain = torch.maximum((power_spec - noise_power) / (power_spec + eps), 
                             torch.tensor(self.wiener_beta))
        
        filtered_D = torch.sqrt(gain) * stft
        y_clean = torch.istft(filtered_D, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        
        return y_clean.unsqueeze(0)
    
    def compress_internal_silences(self, waveform, sample_rate):
        """
        Compresses long internal silence segments using a custom implementation.
        If the detected silence is shorter than the maximum allowed, it is left unchanged.
        """
        waveform = waveform.squeeze()
        
        threshold = 10**(self.silence_threshold_db / -20)
        
        energy = torch.abs(waveform)
        
        is_silence = energy < threshold
        
        max_silence_samples = int(self.max_silence_duration * sample_rate)
        
        silence_starts = []
        silence_ends = []
        
        buffer = 10
        in_silence = False

        for i in range(buffer, len(is_silence) - buffer):
            if not in_silence and is_silence[i]:
                in_silence = True
                silence_starts.append(i)
            elif in_silence and not is_silence[i]:
                in_silence = False
                silence_ends.append(i)

        if in_silence:
            silence_ends.append(len(is_silence) - buffer)
        
        silence_intervals = list(zip(silence_starts, silence_ends))
        
        speech_intervals = []
        audio_length = waveform.shape[0]
        
        if len(silence_intervals) == 0:
            speech_intervals = [[0, audio_length]]
        else:
            current_pos = 0
            for start, end in silence_intervals:
                if current_pos < start:
                    speech_intervals.append([current_pos, start])
                current_pos = end
                
            if current_pos < audio_length:
                speech_intervals.append([current_pos, audio_length])
        
        processed_audio = []
        
        if len(speech_intervals) == 0:
            return waveform.unsqueeze(0)
        
        first_start = speech_intervals[0][0]
        if first_start > 0:
            if first_start > max_silence_samples:
                processed_audio.append(torch.zeros(max_silence_samples, device=waveform.device))
            else:
                processed_audio.append(waveform[:first_start])
        
        for i, (start, end) in enumerate(speech_intervals):
            processed_audio.append(waveform[start:end])
            
            if i < len(speech_intervals) - 1:
                current_end = end
                next_start = speech_intervals[i + 1][0]
                gap = next_start - current_end
                if gap > max_silence_samples:
                    processed_audio.append(torch.zeros(max_silence_samples, device=waveform.device))
                else:
                    processed_audio.append(waveform[current_end:next_start])
        
        last_end = speech_intervals[-1][1]
        if last_end < audio_length:
            trailing_gap = audio_length - last_end
            if trailing_gap > max_silence_samples:
                processed_audio.append(torch.zeros(max_silence_samples, device=waveform.device))
            else:
                processed_audio.append(waveform[last_end:])
        
        compressed_audio = torch.cat(processed_audio)
        
        print(f"Before: {waveform.shape}, After: {compressed_audio.shape}")
        return compressed_audio.unsqueeze(0)
    
    def preprocess_transform(self, waveform, sample_rate):
        """
        Applies comprehensive preprocessing for ASR tasks, including optional volume normalization,
        silence trimming, internal silence compression, high-pass filtering, and noise reduction.
        """
        #(1) Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        #(2) Normalize volume
        effects = [["gain", "-n", "2"]]
        try:
            waveform, sample_rate = apply_effects_tensor(waveform, sample_rate, effects)
        except Exception as e:
            print(f"Volume normalization failed, skipping: {e}")
        
        #(3) Boost volume
        volume_boost = 4.0 
        waveform = waveform * volume_boost
        
        if torch.max(torch.abs(waveform)) > 0.99:
            waveform = waveform / torch.max(torch.abs(waveform)) * 0.99
            print("Applied volume boost with clipping prevention")
        
        #(4) Trim silence end
        try:
            audio_np = waveform.squeeze().numpy()
            trimmed_audio, _ = librosa.effects.trim(audio_np, top_db=20)
            waveform = torch.tensor(trimmed_audio).unsqueeze(0)
            print(f"Trimmed waveform shape: {waveform.shape}")
        except Exception as e:
            print(f"Librosa silence trim failed, skipping: {e}")
        
        #(5) compress long internal silences
        if self.compress_internal:
            try:
                waveform = self.compress_internal_silences(waveform, sample_rate)
                print(f"Compressed internal silences, waveform shape: {waveform.shape}")
            except Exception as e:
                print(f"Compressing internal silences failed, skipping: {e}")
        
        # (6) Apply a lower frequency high-pass filter
        try:
            effects = [["highpass", str(100)]]
            waveform, sample_rate = apply_effects_tensor(waveform, sample_rate, effects)
        except Exception as e:
            print(f"High-pass filter failed, skipping: {e}")
        
        #(7) Apply noise reduction methods
        if self.noise_reduction_methods:
            print(f"Applying noise reduction methods: {self.noise_reduction_methods}")
            for method in self.noise_reduction_methods:
                if method == 'spectral':
                    waveform = self.spectral_subtraction_transform(waveform, sample_rate)
                    print("Applied spectral subtraction")
                elif method == 'wiener':
                    waveform = self.wiener_filter_transform(waveform, sample_rate)
                    print("Applied Wiener filtering")
        
        return waveform

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    
    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None, processed_dir=None, snr_threshold=5, 
                 extreme_noise_threshold=0.2, transcriber=None, transcriptions_dir=None, 
                 move_noisy_files=True, noisy_dir=None):
        """
        Args:
            audio_dir (str): Directory with raw audio files.
            transform (callable, optional): Function to apply preprocessing to the waveform.
            processed_dir (str, optional): Directory to save processed audio files.
            snr_threshold (float): SNR threshold for applying more aggressive processing.
            extreme_noise_threshold (float): SNR threshold below which files are considered 
                                           extremely noisy and may be moved.
            transcriber (WhisperTranscriber, optional): Whisper transcriber instance.
            transcriptions_dir (str, optional): Directory to save transcriptions.
            move_noisy_files (bool): Whether to move extremely noisy files or keep them.
            noisy_dir (str, optional): Directory to move extremely noisy files to.
        """
        self.audio_dir = audio_dir
        self.audio_files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.lower().endswith(('.wav', '.mp3'))
        ]
        self.transform = transform
        self.processed_dir = processed_dir
        self.snr_threshold = snr_threshold
        self.extreme_noise_threshold = extreme_noise_threshold
        self.transcriber = transcriber
        self.transcriptions_dir = transcriptions_dir
        self.move_noisy_files = move_noisy_files
        self.noisy_dir = noisy_dir or os.path.join(os.path.dirname(audio_dir), "audio_noisy")
        
        if self.processed_dir:
            os.makedirs(self.processed_dir, exist_ok=True)
            
        if self.transcriptions_dir:
            os.makedirs(self.transcriptions_dir, exist_ok=True)
        
        if self.move_noisy_files:
            os.makedirs(self.noisy_dir, exist_ok=True)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        try:
            waveform, sample_rate = torchaudio.load(file_path, backend="sox")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            try:
                broken_dir = os.path.join(os.path.dirname(self.audio_dir), "audio_broken")
                os.makedirs(broken_dir, exist_ok=True)
                filename = os.path.basename(file_path)
                
                if os.path.exists(file_path):
                    try:
                        shutil.move(file_path, os.path.join(broken_dir, filename))
                    except Exception as move_error:
                        print(f"Failed to move file {file_path}: {move_error}")
                else:
                    print(f"File already moved by another worker: {file_path}")
                    
               
            except Exception as handle_error:
                print(f"Error handling broken file {file_path}: {handle_error}")
            return None, None, file_path, None
        
        snr = compute_snr(waveform)
        print(f"File: {os.path.basename(file_path)}, SNR: {snr:.2f} dB")
        
        if not hasattr(self, 'snr_values'):
            self.snr_values = []
        self.snr_values.append(snr)
        
        if snr < self.extreme_noise_threshold:
            print(f"File {file_path} has EXTREMELY low SNR: {snr:.2f} dB.")
            
            if self.move_noisy_files:
                print(f"Moving to noisy directory: {self.noisy_dir}")
                try:
                    if os.path.exists(file_path):
                        try:
                            shutil.move(file_path, os.path.join(self.noisy_dir, os.path.basename(file_path)))
                        except Exception as move_error:
                            print(f"Failed to move noisy file {file_path}: {move_error}")
                    else:
                        print(f"Noisy file already moved by another worker: {file_path}")
                        
                   
                except Exception as handle_error:
                    print(f"Error handling noisy file {file_path}: {handle_error}")
                return None, None, file_path, None
            else:
                print("Processing extremely noisy file anyway (move_noisy_files=False)")
        elif snr < self.snr_threshold:
            print(f"File {file_path} has moderate noise (SNR: {snr:.2f} dB). Processing anyway.")
            
        transcription = None
        
        if self.transform:
            waveform = self.transform(waveform, sample_rate)
            
            if idx % 100 == 0: 
                print(f"Processed file: {file_path}")
                print(f"Sample Rate: {sample_rate}")
                print(f"Waveform shape after transform: {waveform.shape}")
            
            waveform_2d = waveform.squeeze()
            
            if waveform_2d.dim() == 1:
                waveform_2d = waveform_2d.unsqueeze(0)
            
            if self.processed_dir:
                output_path = os.path.join(self.processed_dir, os.path.basename(file_path))
                torchaudio.save(output_path, waveform_2d, sample_rate)
            
            if self.transcriber:
                transcription = self.transcriber.transcribe_file(file_path)
                
                if self.transcriptions_dir:
                    base_name = Path(file_path).stem
                    txt_path = os.path.join(self.transcriptions_dir, f"{base_name}.txt")
                    
                    with open(txt_path, 'w') as f:
                        f.write(transcription["text"])
                
                print(f"Transcribed: {transcription['text'][:50]}...")
                
            waveform = waveform_2d
        
        if hasattr(self, 'snr_values'):
            snrs = self.snr_values
            print(f"\nSNR Statistics:")
            print(f"Min SNR: {min(snrs):.2f} dB")
            print(f"Max SNR: {max(snrs):.2f} dB")
            print(f"Mean SNR: {sum(snrs)/len(snrs):.2f} dB")
            print(f"Histogram: {np.histogram(snrs, bins=[0, 2, 5, 10, 15, 20, 30, 100])}")
        
        return waveform, sample_rate, file_path, transcription
    
def custom_collate_fn(batch):
    """
    Custom collate function that handles None values and pads sequences.
    
    Args:
        batch: A batch of samples from the dataset
        
    Returns:
        tuple: (padded_waveforms, lengths, sample_rates, file_paths, transcriptions)
    """
    batch = [item for item in batch if item[0] is not None]
    
    if len(batch) == 0:
        return None, None, None, None, None
    
    waveforms, sample_rates, file_paths, transcriptions = zip(*batch)
    
    padded_waveforms, lengths = pad_sequence(waveforms)
    
    return padded_waveforms, lengths, torch.tensor(sample_rates), file_paths, transcriptions

def pad_sequence(batch, padding_value=0.0):
    """
    Pads a batch of variable length audio tensors to the same length.
    
    Args:
        batch (list): List of audio tensors [batch_size, channels, time]
        padding_value (float): Value to pad with
        
    Returns:
        Tensor: Padded tensor [batch_size, channels, max_time]
    """
    lengths = [x.shape[-1] for x in batch]
    max_len = max(lengths)
    
    batch_size = len(batch)
    n_channels = batch[0].shape[0]
    
    padded = torch.full((batch_size, n_channels, max_len), padding_value)
    
    for i, tensor in enumerate(batch):
        length = tensor.shape[-1]
        padded[i, :, :length] = tensor
        
    return padded, torch.tensor(lengths)

def load_environment():
    """Load environment variables from .env file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
        if os.getenv('OPENAI_API_KEY'):
            print("OPENAI_API_KEY found in environment variables")
        else:
            print("WARNING: OPENAI_API_KEY not found in .env file")
    else:
        print(f"No .env file found at {env_path}")

class OpenAITranscriber:
    def __init__(self, model="whisper-1", language="en", prompt=None):
        """
        Initializes the OpenAI transcriber.
        
        Args:
            model (str): The OpenAI model to use for transcription
            language (str): The language code (e.g., "en", "es")
            prompt (str, optional): A prompt to guide the transcription
        """
        self.client = OpenAI()
        self.model = model
        self.language = language
        self.prompt = prompt
        
    def transcribe_file(self, audio_path):
        """
        Transcribes a single audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: The transcription result
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    language=self.language,
                    prompt=self.prompt
                )
            return {"text": transcription.text}
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return {"text": "", "error": str(e)}

if __name__ == "__main__":
    load_environment()
    
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    AUDIO_DIR = "./audio_raw"
    PROCESSED_DIR = "./audio_processed"
    
    BATCH_SIZE = 8
    
    transform = AudioTransform(
        noise_reduction_methods=['spectral', 'wiener'],
        compress_internal=True,
        silence_threshold_db=20,
        max_silence_duration=0.005,
        wiener_beta=0.005, 
        highpass_cutoff=100,  
        device=device
    )
    
    dataset = AudioDataset(
        audio_dir=AUDIO_DIR, 
        transform=transform,
        processed_dir=PROCESSED_DIR,
        snr_threshold=5.0,           
        extreme_noise_threshold=2.0, 
        transcriber=None,
        transcriptions_dir=None,
        move_noisy_files=True,       
        noisy_dir="./audio_noisy"    
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=2
    )
    
    transcriber = OpenAITranscriber(
        model="gpt-4o-transcribe",
        language="es",
        prompt="Actua como un radiologo que traduce audio de radiologia."
    )
    
    TRANSCRIPTIONS_DIR = "./transcriptions"
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    
    print("\n=== Processing Audio and Transcribing ===")
    for i, batch in enumerate(tqdm(dataloader, desc="Processing Audio")):
        padded_waveforms, lengths, sample_rates, file_paths, _ = batch
        
        if padded_waveforms is None:
            continue
            
        padded_waveforms = padded_waveforms.to(device)
        
        for file_path in file_paths:
            processed_path = os.path.join(PROCESSED_DIR, os.path.basename(file_path))
            if os.path.exists(processed_path):
                result = transcriber.transcribe_file(processed_path)
                
                base_name = Path(file_path).stem
                txt_path = os.path.join(TRANSCRIPTIONS_DIR, f"{base_name}.txt")
                
                with open(txt_path, 'w') as f:
                    f.write(result["text"])
                
                print(f"Transcribed: {os.path.basename(file_path)} - {result['text'][:50]}...")
            else:
                print(f"WARNING: Processed file does not exist: {os.path.basename(processed_path)}")
    
    print("\nAll processing and transcription complete!")