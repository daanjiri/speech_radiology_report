# Medical Speech Transcription & Summarization Pipeline 🏥🎙️

## Project Overview

This project implements an end-to-end AI pipeline for processing medical dictations, automatically transcribing audio recordings from radiologists and generating concise, accurate summaries of the medical reports. The system was developed as a solution for healthcare professionals to streamline their documentation workflow.

## 🎯 Problem Statement

Medical professionals, particularly radiologists, spend significant time dictating findings and creating reports. This project addresses the need for:
- Accurate transcription of medical terminology from Spanish audio
- Intelligent summarization that preserves critical medical information
- Scalable cloud architecture for healthcare institutions

## 🚀 Key Features

### 1. **Advanced Audio Processing**
- Automatic detection and filtering of corrupted or excessively noisy recordings
- Multi-method noise reduction (Spectral Subtraction, Wiener Filtering)
- Audio quality assessment using Signal-to-Noise Ratio (SNR) metrics
- Intelligent silence removal and audio compression

### 2. **Medical-Specific Transcription**
- Integration with OpenAI's GPT-4o-Transcribe model
- Custom prompts optimized for radiology terminology
- Batch processing capabilities for multiple audio files
- Error handling for problematic recordings

### 3. **Multi-Model Summarization Comparison**
- **BERT-based Models**: BERT2BERT fine-tuned for Spanish medical text
- **Large Language Models**: Deepseek-R1 for advanced reasoning
- **Multilingual Models**: mBART for cross-language capabilities
- Comprehensive evaluation using multiple metrics

### 4. **Production-Ready Cloud Architecture**
- Serverless design using AWS Lambda for cost efficiency
- Asynchronous processing with Amazon SQS
- Scalable storage with S3 and DynamoDB
- API Gateway for secure, scalable access

## 📊 Technical Implementation

### Models Evaluated

| Model | Type | Strengths |
|-------|------|-----------|
| BERT2BERT | Transformer | Fast, lightweight, good for Spanish text |
| mBART-large | Multilingual | Cross-language capabilities |
| Deepseek-R1 | LLM | Superior reasoning and context understanding |

### Evaluation Metrics

The system evaluates summaries using:
- **ROUGE Scores** (1, 2, L): N-gram overlap analysis
- **METEOR**: Translation quality with synonym matching
- **BERTScore**: Semantic similarity using contextual embeddings
- **Content Coverage**: Preservation of important medical terms
- **Compression Ratio**: Summary conciseness
- **Abstractiveness**: Novel wording generation

## 🏗️ System Architecture

The production architecture leverages AWS services for scalability:

```
Audio Input → API Gateway → Lambda (Transcription) → S3/DynamoDB
                                ↓
                              SQS Queue
                                ↓
                     Bedrock/Groq (Summarization)
                                ↓
                          DynamoDB → API Response
```

### Architecture Highlights
- **Serverless**: Pay-per-use model with automatic scaling
- **Decoupled Components**: Queue-based processing for reliability
- **HIPAA Considerations**: Data privacy with no-retention policies
- **High Availability**: Multi-AZ deployment capabilities

## 💻 Local Development

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- OpenAI API key for transcription

### Installation

```bash
git clone https://github.com/daanjiri/speech_radiology_report.git
cd speech_radiology_report
pip install -r requirements.txt
```

### Usage

1. **Audio Preprocessing & Transcription**
   ```bash
   python preprocess.py
   ```
   - Processes audio files from `./audio_raw/`
   - Applies noise reduction and quality filtering
   - Generates transcriptions in `./transcriptions/`

2. **Batch Summarization**
   ```bash
   python batch_summarization.py
   ```
   - Processes all transcriptions
   - Compares multiple summarization models
   - Outputs results to `./summarization_results/`

3. **View Metrics**
   ```bash
   python metrics.py
   ```
   - Displays comprehensive evaluation metrics
   - Identifies best-performing models

## 📁 Project Structure

```
speech_radiology_report/
├── preprocess.py              # Audio processing & ASR pipeline
├── batch_summarization.py     # Multi-model summarization engine
├── metrics.py                 # Evaluation metrics implementation
├── summary.py                 # Individual summarization utilities
├── requirements.txt           # Python dependencies
├── arquitectura/              # Cloud architecture diagrams
│   ├── diagrama_arquitectura.png
│   └── diagrama_arquitectura.txt
├── audio_raw/                 # Input audio files
├── audio_processed/           # Cleaned audio files
├── audio_noisy/              # Rejected noisy files
├── transcriptions/           # ASR output texts
└── summarization_results/    # Generated summaries & metrics
```

## 🔬 Results & Findings

Based on extensive testing with medical dictations:

1. **Best Overall Model**: Deepseek-R1 demonstrated superior performance in preserving medical context and reasoning
2. **Transcription Accuracy**: GPT-4o-Transcribe outperformed Whisper for Spanish medical terminology
3. **Processing Efficiency**: BERT2BERT offers the best speed-to-quality ratio for real-time applications
4. **Noise Handling**: Combined spectral and Wiener filtering improved transcription accuracy by 23%

Detailed metrics and model comparisons are available in `summarization_results/best_model.txt`.

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, Transformers, BERTScore
- **Audio Processing**: Librosa, SciPy, NumPy
- **NLP**: NLTK, Rouge-Score, spaCy
- **Cloud Services**: AWS Lambda, S3, DynamoDB, SQS, API Gateway
- **APIs**: OpenAI (GPT-4o), Groq, Amazon Bedrock


## 🤝 Contact

For questions or collaboration opportunities, please reach out through my portfolio or GitHub.

---

*This project demonstrates expertise in audio processing, natural language processing, cloud architecture, and healthcare technology integration.*