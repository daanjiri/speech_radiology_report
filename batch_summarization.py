import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import create_summarizer, evaluate_summary, models

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

def batch_summarize():
    # Create dataset and dataloader
    dataset = TranscriptionDataset('transcriptions')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Dictionary to store metrics for each model
    all_metrics = {model_name: [] for model_name in models.keys()}
    
    # Create output directory
    output_dir = 'summarization_results'
    ensure_dir(output_dir)
    
    # Create metrics directory for individual CSV files
    metrics_dir = os.path.join(output_dir, 'metrics')
    ensure_dir(metrics_dir)
    
    # Create summarizers for each model
    summarizers = {}
    for model_name, model_id in models.items():
        print(f"Loading {model_name}...")
        summarizers[model_name] = create_summarizer(model_id)
        
        # Create model-specific output directory
        model_dir = os.path.join(output_dir, model_name)
        ensure_dir(model_dir)
    
    # Get total number of files for progress tracking
    total_files = len(dataset)
    print(f"Starting to process {total_files} files with {len(summarizers)} models...")
    
    # Process each file with a tqdm progress bar
    for file_idx, batch in enumerate(tqdm(dataloader, desc="Files processed", position=0)):
        text = batch['text'][0]  # Batch size is 1
        file_name = batch['file_name'][0]
        
        # Skip empty texts
        if not text:
            print(f"Skipping empty file: {file_name}")
            continue
        
        # Process with each model with a nested tqdm progress bar
        for model_name in tqdm(summarizers.keys(), desc=f"File {file_idx+1}/{total_files}: {file_name}", position=1, leave=False):
            summarizer = summarizers[model_name]
            if summarizer is None:
                continue
                
            try:
                # Generate summary
                summary_output = summarizer(text, max_length=150, min_length=50, do_sample=False)
                summary = summary_output[0]['summary_text']
                
                # Save summary to file
                output_file = os.path.join(output_dir, model_name, file_name)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                # Evaluate summary
                metrics = evaluate_summary(text, summary)
                metrics['file_name'] = file_name
                all_metrics[model_name].append(metrics)
                
            except Exception as e:
                print(f"Error processing {file_name} with {model_name}: {e}")
    
    # Clear the progress bars
    print("\nAll files processed. Calculating metrics...")
    
    # Save individual metrics for each model to separate CSV files
    for model_name in tqdm(all_metrics.keys(), desc="Saving individual metrics"):
        metrics_list = all_metrics[model_name]
        if not metrics_list:
            print(f"No successful summaries for {model_name}")
            continue
            
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Save individual metrics to CSV
        model_metrics_file = os.path.join(metrics_dir, f"{model_name}_metrics.csv")
        df.to_csv(model_metrics_file, index=False)
        print(f"Saved individual metrics for {model_name} to {model_metrics_file}")
        
        # Calculate average metrics
        avg_metrics = df.drop(columns=['file_name']).mean()
        all_metrics[model_name] = avg_metrics
    
    # Create comparison DataFrame for global metrics
    df_global = pd.DataFrame({model: metrics for model, metrics in all_metrics.items() 
                            if not isinstance(metrics, list)}).T
    
    # Save global metrics to CSV
    global_metrics_file = os.path.join(output_dir, 'global_metrics.csv')
    df_global.to_csv(global_metrics_file)
    print(f"Saved global metrics to {global_metrics_file}")
    
    # Display global metrics
    print("\n=== GLOBAL METRICS COMPARISON ===")
    pd.set_option('display.precision', 4)
    print(df_global)
    
    # Find best model for each metric
    print("\n=== BEST MODELS PER METRIC ===")
    metric_cols = ["rouge1", "rouge2", "rougeL", "meteor", "content_coverage"]
    if "bertscore_f1" in df_global.columns:
        metric_cols.append("bertscore_f1")
    
    for metric in metric_cols:
        if metric in df_global.columns:
            best_model = df_global[metric].idxmax()
            best_score = df_global.loc[best_model, metric]
            print(f"Best {metric}: {best_model} ({best_score:.4f})")
    
    # Best compression (closest to 0.3, which is a common target)
    if "compression_ratio" in df_global.columns:
        best_compression = (df_global["compression_ratio"] - 0.3).abs().idxmin()
        compression_score = df_global.loc[best_compression, "compression_ratio"]
        print(f"Best compression ratio: {best_compression} ({compression_score:.4f})")
    
    # Fastest model
    if "generation_time" in df_global.columns:
        fastest = df_global["generation_time"].idxmin()
        fastest_time = df_global.loc[fastest, "generation_time"]
        print(f"Fastest model: {fastest} ({fastest_time:.4f} seconds)")

if __name__ == "__main__":
    batch_summarize() 