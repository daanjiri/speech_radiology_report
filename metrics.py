import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import warnings
import nltk
from nltk.translate.meteor_score import meteor_score
warnings.filterwarnings("ignore")
# Download NLTK resources needed for METEOR
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet

# List of models to compare - Move this to the top level of the file so it can be imported
models = {
    "BERT2BERT": "mrm8488/bert2bert_shared-spanish-finetuned-summarization",
    "mBART-large": "facebook/mbart-large-cc25",
    # "mT5-base": "google/flan-t5-large"
}

# Function to create summarization pipeline
def create_summarizer(model_name, device="cpu"):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

# METEOR score calculation
def calculate_meteor(reference, hypothesis):
    # Split into tokens (METEOR expects lists of tokens)
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    # Calculate METEOR score
    # Note: For Spanish, wordnet support might be limited,
    # so some synonym matching aspects of METEOR might not work as well
    try:
        score = meteor_score([reference_tokens], hypothesis_tokens)
        return score
    except Exception as e:
        print(f"METEOR calculation error: {e}")
        return 0.0

# Define evaluation metrics
def evaluate_summary(original_text, summary):
    results = {}
    
    # 1. ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, summary)
    results["rouge1"] = scores['rouge1'].fmeasure
    results["rouge2"] = scores['rouge2'].fmeasure
    results["rougeL"] = scores['rougeL'].fmeasure
    
    # 2. METEOR Score
    results["meteor"] = calculate_meteor(original_text, summary)
    
    # 3. Content coverage
    vectorizer = TfidfVectorizer()
    vectorizer.fit([original_text])
    original_vec = vectorizer.transform([original_text]).toarray()[0]
    summary_vec = vectorizer.transform([summary]).toarray()[0]
    important_indices = np.nonzero(original_vec)[0]
    coverage = sum(1 for idx in important_indices if summary_vec[idx] > 0)
    results["content_coverage"] = coverage / len(important_indices) if len(important_indices) > 0 else 0
    
    # 4. Compression ratio
    results["compression_ratio"] = len(summary.split()) / len(original_text.split())
    
    return results