# DistilBERT Sentiment Analysis for Amazon Reviews

A simple, transformer-based Natural Language Processing (NLP) project that fine-tunes a DistilBERT model to classify Amazon product reviews as either **Positive** or **Negative**. 

## Overview
This project uses the Hugging Face `transformers` and `datasets` libraries to train a sequence classification model. It merges the title and content of Amazon reviews to maximize contextual signal, tokenizes the text, and trains the model using the Hugging Face `Trainer` API.

## Features
* **Model:** `distilbert-base-uncased` (a faster, lighter version of BERT).
* **Dataset:** `amazon_polarity` (30,000 training samples, 5,000 validation samples, 3,000 test samples).
* **Metrics Tracked:** Accuracy, F1-Score, Precision, and Recall.
* **Inference:** Includes a custom function to test new, unseen reviews and output confidence scores.

## Prerequisites
Ensure you have Python installed, along with the following dependencies. You can install them via pip:

```bash
pip install torch numpy datasets transformers scikit-learn
