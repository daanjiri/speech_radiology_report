import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import create_summarizer, evaluate_summary, models
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class TranscriptionDataset(Dataset):
    """Dataset for loading transcription files."""
    
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder containing transcription files
        """
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        print(f"Found {len(self.file_list)} text files in {folder_path}")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return {
                'text': text,
                'file_name': file_name
            }
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return {
                'text': '',
                'file_name': file_name
            }

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_groq_summary(text, model_id):
    """Generate summary using Groq API."""
    try:
        groq_model = model_id.replace("api:", "")
        
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """Eres un asistente de radiologia que resume estudios o observaciones de radiologia.
                    tenga encuenta que las observaciones son de audios transcritos entonces puede que tengan errores de transcripción.
                    Solo dame una frase corta y concisa que resuma la observación.
                    """,
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model=groq_model,
            stream=False,
            temperature=0,
        )
        
        summary = chat_completion.choices[0].message.content
        
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()
            
        return summary
    except Exception as e:
        print(f"Error generating summary with Groq API: {e}")
        return ""

def batch_summarize():
    dataset = TranscriptionDataset('transcriptions')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_metrics = {model_name: [] for model_name in models.keys()}
    
    output_dir = 'summarization_results'
    ensure_dir(output_dir)
    
    metrics_dir = os.path.join(output_dir, 'metrics')
    ensure_dir(metrics_dir)
    
    metric_files = {}
    for model_name in models.keys():
        model_metrics_file = os.path.join(metrics_dir, f"{model_name}_metrics.csv")
        if not os.path.exists(model_metrics_file):
            pd.DataFrame(columns=['file_name', 'rouge1', 'rouge2', 'rougeL', 
                                 'meteor', 'content_coverage', 'compression_ratio', 
                                 'bert_score_f1', 'bert_score_precision', 'bert_score_recall']).to_csv(model_metrics_file, index=False)
        metric_files[model_name] = model_metrics_file
    
    summarizers = {}
    for model_name, model_id in models.items():
        print(f"Loading {model_name}...")
        if not model_id.startswith("api:"):
            summarizers[model_name] = create_summarizer(model_id, device=device)
        else:
            summarizers[model_name] = "api"
        
        model_dir = os.path.join(output_dir, model_name)
        ensure_dir(model_dir)
    
    total_files = len(dataset)
    print(f"Starting to process {total_files} files with {len(summarizers)} models...")
    
    for file_idx, batch in enumerate(tqdm(dataloader, desc="Files processed", position=0)):
        text = batch['text'][0]  # Batch size is 1
        file_name = batch['file_name'][0]
        
        if not text:
            print(f"Skipping empty file: {file_name}")
            continue
        
        for model_name in tqdm(summarizers.keys(), desc=f"File {file_idx+1}/{total_files}: {file_name}", position=1, leave=False):
            summarizer = summarizers[model_name]
            if summarizer is None:
                continue
                
            try:
                if summarizer == "api":
                    model_id = models[model_name]
                    if "deepseek" in model_id.lower():
                        summary = generate_groq_summary(text, model_id)
                    else:
                        print(f"Unsupported API model: {model_id}")
                        continue
                else:
                    if "mbart" in model_name.lower():
                        summary = summarizer(
                            text, 
                            max_length=150, 
                            min_length=30,
                            no_repeat_ngram_size=3,
                            num_beams=5,
                            length_penalty=1.0,
                            early_stopping=True,
                            forced_bos_token_id=summarizer.tokenizer.lang_code_to_id["es_XX"]  # Spanish token ID
                        )
                    elif "bert2bert" in model_name.lower():
                        summary = summarizer(
                            text, 
                            max_length=150, 
                            min_length=30,
                            no_repeat_ngram_size=3,
                            num_beams=5,
                            length_penalty=1.0,
                            early_stopping=True
                        )
                    else:
                        summary = summarizer(
                            text, 
                            max_length=150, 
                            min_length=30,
                            no_repeat_ngram_size=3,
                            num_beams=5,
                            length_penalty=1.0
                        )
                    
                    if isinstance(summary, str):
                        pass
                    elif isinstance(summary, list) and isinstance(summary[0], dict):
                        summary = summary[0]['summary_text']
                    elif isinstance(summary, list):
                        summary = summary[0]
                
                output_file = os.path.join(output_dir, model_name, file_name)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                metrics = evaluate_summary(text, summary)
                metrics['file_name'] = file_name
                
                all_metrics[model_name].append(metrics)
                
                metrics_df = pd.DataFrame([metrics])
                
                available_columns = ['file_name']
                for col in ['rouge1', 'rouge2', 'rougeL', 'meteor', 'content_coverage', 
                           'compression_ratio', 'bert_score_f1',
                           'bert_score_precision', 'bert_score_recall']:
                    if col in metrics:
                        available_columns.append(col)
                
                metrics_df = metrics_df[available_columns]
                metrics_df.to_csv(metric_files[model_name], mode='a', header=False, index=False)
                
            except Exception as e:
                print(f"Error processing {file_name} with {model_name}: {e}")
    
    print("\nAll files processed. Calculating metrics...")
    
    global_avg_metrics = {}
    
    for model_name in tqdm(all_metrics.keys(), desc="Calculating global metrics"):
        model_metrics_file = metric_files[model_name]
        if os.path.exists(model_metrics_file):
            df = pd.read_csv(model_metrics_file)
            if len(df) > 0:
                if 'file_name' in df.columns:
                    metric_columns = [col for col in df.columns if col != 'file_name']
                    avg_metrics = df[metric_columns].mean()
                    global_avg_metrics[model_name] = avg_metrics
            else:
                print(f"No successful summaries for {model_name}")
        else:
            print(f"No metrics file found for {model_name}")
    
    df_global = pd.DataFrame(global_avg_metrics).T
    
    global_metrics_file = os.path.join(output_dir, 'global_metrics.csv')
    df_global.to_csv(global_metrics_file)
    print(f"Saved global metrics to {global_metrics_file}")
    
    print("\n=== GLOBAL METRICS COMPARISON ===")
    pd.set_option('display.precision', 4)
    print(df_global)
    
    print("\n=== BEST MODELS PER METRIC ===")
    metric_cols = ["rouge1", "rouge2", "rougeL", "meteor", "content_coverage", 
                  "bert_score_f1", "bert_score_precision", "bert_score_recall"]
    
    for metric in metric_cols:
        if metric in df_global.columns:
            best_model = df_global[metric].idxmax()
            best_score = df_global.loc[best_model, metric]
            print(f"Best {metric}: {best_model} ({best_score:.4f})")
    
    if "compression_ratio" in df_global.columns:
        best_compression = (df_global["compression_ratio"] - 0.3).abs().idxmin()
        compression_score = df_global.loc[best_compression, "compression_ratio"]
        print(f"Best compression ratio: {best_compression} ({compression_score:.4f})")
    
    if "generation_time" in df_global.columns:
        fastest = df_global["generation_time"].idxmin()
        fastest_time = df_global.loc[fastest, "generation_time"]
        print(f"Fastest model: {fastest} ({fastest_time:.4f} seconds)")

if __name__ == "__main__":
    batch_summarize() 