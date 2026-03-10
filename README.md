# Text summarization with a Transformer from scratch

This repo contains a Kaggle notebook where I build an encoder–decoder Transformer from scratch in TensorFlow/Keras and train it for news headline generation on the Inshorts dataset.

The goal is educational: implement the core pieces of a Transformer (multi‑head attention, masks, positional encoding, custom learning‑rate schedule) and run a full end‑to‑end text summarization pipeline.

---

## What it does

- Loads and cleans the Inshorts news dataset (around 55k articles and headlines).  
- Applies preprocessing and word‑level tokenization to create input (article) and target (summary) sequences.  
- Builds an encoder–decoder Transformer with 3 layers, 4 heads, model dimension 256, feed‑forward dimension 512, dropout, for roughly 48M parameters.  
- Trains the model for 50 epochs with a warmup‑based learning‑rate schedule and tracks loss and token‑level accuracy.  
- Evaluates the model with ROUGE on the training data and inspects generated summaries qualitatively.

---

## Tech stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  
- `rouge-score` for ROUGE evaluation  

---

## Notebook structure

Main notebook:

- `text-summarization-transformers-from-scratch.ipynb`

High‑level sections:

1. **Seq2Seq and Transformer recap**  
   Short theory section on encoder–decoder models, attention and tokenization.

2. **Data loading and preprocessing**  
   - Load the Inshorts dataset from Excel  
   - Clean text (lowercasing, remove special characters and URLs)  
   - Limit article length (e.g. 100 tokens) and summary length (e.g. 15 tokens)  
   - Add start/end tokens to summaries and build vocabularies

3. **Tokenization and batching**  
   - Map tokens to integer IDs  
   - Pad/truncate sequences to fixed length  
   - Create batched TensorFlow datasets for training

4. **Transformer implementation**  
   - Positional encoding  
   - Scaled dot‑product attention and multi‑head attention  
   - Encoder and decoder blocks (self‑attention, cross‑attention, feed‑forward, residual connections, layer norm)  
   - Full encoder–decoder Transformer model

5. **Training loop**  
   - Custom learning‑rate schedule with warmup  
   - Training step with teacher forcing and masking  
   - 50‑epoch training run with logged loss and accuracy per epoch

6. **Evaluation and examples**  
   - Compute ROUGE‑1, ROUGE‑2, ROUGE‑L on the training data  
   - Show input → generated summary examples
