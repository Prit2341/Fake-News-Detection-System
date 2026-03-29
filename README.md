---
title: Fake News Detection
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
---

# Fake News Detection — MLOps + ML Showcase

Multilingual ensemble model combining **LightGBM** and **XLM-RoBERTa** to detect fake news with **99.71% accuracy**.

## Architecture
- **Sentence-BERT** (`paraphrase-multilingual-MiniLM-L12-v2`) generates 384-dim multilingual embeddings
- **LightGBM** (500 trees, GPU-trained) classifies on embeddings
- **XLM-RoBERTa** fine-tuned with gradient checkpointing + AMP for 4GB VRAM
- **Ensemble** with grid-searched optimal weights

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 99.71% |
| F1 Macro | 99.71% |
| AUC-ROC | 0.9999 |

## Tech Stack
PyTorch · HuggingFace Transformers · LightGBM · Sentence-Transformers · FastAPI · PostgreSQL
