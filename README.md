# Deep Learning NLP Text Classifier — Spam Filter (PyTorch)

A feed-forward neural network in PyTorch that classifies short text messages as spam or
ham (not spam): TF-IDF vectorization, Dropout + Batch Normalization, a training loop
with manual early stopping, loss/accuracy convergence curves, and inference on unseen
example messages with confidence scores.

## Contents

| File | Purpose |
|---|---|
| [`nlp_text_classifier.ipynb`](nlp_text_classifier.ipynb) | Full pipeline: data, vectorization, model, training, curves, inference |
| `requirements.txt` | Python dependencies |

## Approach

- **Dataset:** synthetic, template-generated spam/ham messages (no real dataset was
  supplied for this exercise). Swap section 2 for a real labeled text dataset — nothing
  downstream needs to change as long as you produce a `texts` list and a `labels` list
  of 0/1.
- **Vectorization:** TF-IDF (1- and 2-grams, 500 features), fit only on the training
  split. A commented-out Hugging Face `sentence-transformers` alternative is included in
  section 3 if you have internet access and want richer embeddings instead.
- **Model:** 2 hidden layers, each `Linear → BatchNorm1d → ReLU → Dropout(0.3)`, single
  output logit trained with `BCEWithLogitsLoss`.
- **Early stopping:** implemented by hand (PyTorch has no built-in callback system like
  Keras) — training stops once validation loss hasn't improved for a set number of
  epochs, and the best-seen weights are restored.
- **Proof included:** loss/accuracy convergence curves (section 7), test-set metrics
  (section 8), and predictions with confidence scores on brand-new example messages
  (section 9).

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook nlp_text_classifier.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All). Training is quick (small
model, small dataset) — a couple of minutes on CPU.
