# Speech Processing and Summarization Pipeline

This project provides a pipeline for processing audio files with Automatic Speech Recognition (ASR) and generating concise summaries of the transcribed content.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/speech-summarization.git
cd speech-summarization
pip install -r requirements.txt
```

## Usage Instructions

### Step 1: Preprocessing (ASR)

Run the preprocessing script to convert audio files to text using Automatic Speech Recognition:

```bash
python preprocess.py
```

This script will:

- Process audio files from the input directory
- Apply speech recognition to generate transcripts
- Save the processed text for the next step

### Step 2: Batch Summarization

After preprocessing is complete, run the batch summarization script:

```bash
python batch_summarization.py
```

This will generate concise summaries for all processed text files.

## Project Structure

```
speech-summarization/
├── preprocess.py          # ASR processing script
├── batch_summarization.py # Text summarization script
├── requirements.txt       # Required dependencies
├── audio_raw/                 # Directory for audio files
├── transcripts/           # Directory for ASR output
├── audio_processed/           # Directory for processed audio files
└── summarization_results/             # Directory for generated summaries
```

## Findings

file: Best_model.txt
path: summarization_results/best_model.txt

This file contains the best model for summarization and the observations about the results.

folder: arquitectura
path: arquitectura/diagrama_arquitectura.txt
path: arquitectura/diagrama_arquitectura.png

This folder contains the diagram of the architecture of the project and the explanation of the architecture.
