import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import warnings
import nltk
from nltk.translate.meteor_score import meteor_score
import bert_score
warnings.filterwarnings("ignore")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  

models = {
    "BERT2BERT": "mrm8488/bert2bert_shared-spanish-finetuned-summarization",
    "mBART-large": "facebook/mbart-large-cc25",
    "Deepseek": "api:deepseek-r1-distill-llama-70b" 
}

def create_summarizer(model_name, device="cpu"):
    """
    Creates a summarization pipeline using the specified model.
    
    This function loads a pre-trained summarization model from Hugging Face's
    model repository and configures it for text summarization tasks. It handles
    special configuration for specific model types like mBART.
    
    Parameters:
        model_name (str): Name of the model to load, either a Hugging Face model ID
                          or an API model (prefixed with "api:").
        device (str, optional): Device to run the model on ('cpu', 'cuda', or specific GPU index).
                                Defaults to "cpu".
    
    Returns:
        pipeline or None: A configured summarization pipeline if successful,
                          None if the model is an API model or if loading fails.
    
    Note:
        For API-based models (starting with "api:"), this function returns None
        as these models are handled separately through API calls.
    """
    if model_name.startswith("api:"):
        return None
        
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if "mbart" in model_name.lower():
            tokenizer.src_lang = "es_XX"
            
        return pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def calculate_meteor(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    try:
        score = meteor_score([reference_tokens], hypothesis_tokens)
        return score
    except Exception as e:
        print(f"METEOR calculation error: {e}")
        return 0.0

def evaluate_summary(original_text, summary):
    """
    Evaluates the quality of a summary using multiple metrics.
    
    This function compares a generated summary against the original text
    using various NLP evaluation metrics to assess quality, coverage,
    and similarity.
    
    Parameters:
        original_text (str): The source text that was summarized
        summary (str): The generated summary to evaluate
        
    Returns:
        dict: Dictionary containing the following evaluation metrics:
            - rouge1, rouge2, rougeL: ROUGE precision/recall/F1 scores
            - meteor: METEOR score for translation quality
            - content_coverage: Percentage of important content preserved
            - compression_ratio: Length ratio of summary to original text
            - bert_score_precision: BERTScore precision
            - bert_score_recall: BERTScore recall
            - bert_score_f1: BERTScore F1
            - abstractiveness: Percentage of summary tokens not in original text
    """
    results = {}
    
    # 1. ROUGE Scores - measures n-gram overlap between summary and original
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, summary)
    results["rouge1"] = scores['rouge1'].fmeasure  # Unigram overlap
    results["rouge2"] = scores['rouge2'].fmeasure  # Bigram overlap
    results["rougeL"] = scores['rougeL'].fmeasure  # Longest common subsequence
    
    # 2. METEOR Score - evaluates translation quality with synonyms and stemming
    results["meteor"] = calculate_meteor(original_text, summary)
    
    # 3. Content coverage - measures how well important terms from original are preserved
    vectorizer = TfidfVectorizer()
    vectorizer.fit([original_text])
    original_vec = vectorizer.transform([original_text]).toarray()[0]
    summary_vec = vectorizer.transform([summary]).toarray()[0]
    important_indices = np.nonzero(original_vec)[0]
    coverage = sum(1 for idx in important_indices if summary_vec[idx] > 0)
    results["content_coverage"] = coverage / len(important_indices) if len(important_indices) > 0 else 0
    
    # 4. Compression ratio - indicates summary conciseness (lower is more concise)
    results["compression_ratio"] = len(summary.split()) / len(original_text.split())
    
    # 5. BERT Score - semantic similarity using contextual embeddings
    P, R, F1 = bert_score.score([summary], [original_text], lang='es', verbose=False)
    results["bert_score_precision"] = P.item()  # How much of summary content is in the original
    results["bert_score_recall"] = R.item()     # How much of original content is in the summary
    results["bert_score_f1"] = F1.item()        # Harmonic mean of precision and recall
    
    # 6. Abstractiveness Score - measures how much the summary uses novel wording
    original_tokens = set(original_text.lower().split())
    summary_tokens = set(summary.lower().split())
    novel_tokens = len(summary_tokens - original_tokens)
    results["abstractiveness"] = novel_tokens / len(summary_tokens) if summary_tokens else 0
    
    return results