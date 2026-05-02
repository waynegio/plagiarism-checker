# Plagiarism Checker

A web-based plagiarism checker built with Python and Streamlit using the MSR Paraphrase Corpus dataset. This project analyzes whether two sentences are paraphrases (have similar meaning) or not.

**Live Demo:** [https://paircheck.streamlit.app/](https://paircheck.streamlit.app/)

---

## Installation Guide

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd plagiarism-checker
```

> **Requirements:** Python 3.13+

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run Home.py
```

Open the link shown in the terminal (usually **http://localhost:8501**) in your browser.

---

## Project Structure

| File / Folder                | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| `Home.py`                    | Main entry point for the Streamlit app                             |
| `model/train.py`             | Training pipeline for the plagiarism model                         |
| `model/plagiarism_model.pkl` | Trained model (auto-generated after training)                      |
| `model/logistic_model.pkl`   | Logistic Regression model for similarity check                     |
| `preprocessing.py`           | Feature extraction and text cleaning functions                     |
| `MSRParaphraseCorpus/`       | Dataset containing training and test data                          |
| `pages/`                     | Additional Streamlit pages (model training UI, similarity checker) |

---

## Features

- **7 Similarity Features**: Cosine TF-IDF, Jaccard, Levenshtein, 3-gram, S-BERT, Word2Vec, GloVe
- **Logistic Regression** classifier for paraphrase detection
- **Interactive UI** with parameter controls and real-time similarity checking
- **Model evaluation** with accuracy, classification report, and confusion matrix
