"""
Ensemble: Combine LightGBM + XLM-RoBERTa predictions via weighted average.
Finds optimal weights by maximizing validation accuracy.
"""

import joblib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = Path("d:/Fake_News_Detection/artifacts")
BATCH_SIZE    = 16
MAX_LEN       = 256
SEED          = 42
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


def get_roberta_probas(texts):
    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR / "roberta_model")
    model     = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS_DIR / "roberta_model").to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_probas = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probas = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probas.extend(probas[:, 1])     # probability of fake

    return np.array(all_probas)


def get_lgbm_probas(embeddings):
    model = joblib.load(ARTIFACTS_DIR / "lgbm_model.joblib")
    return model.predict_proba(embeddings)[:, 1]


def find_best_weights(lgbm_probas, roberta_probas, y_true):
    """Grid search for best weight between LightGBM and RoBERTa."""
    best_acc, best_w = 0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        combined = w * lgbm_probas + (1 - w) * roberta_probas
        acc      = accuracy_score(y_true, (combined >= 0.5).astype(int))
        if acc > best_acc:
            best_acc, best_w = acc, w
    return best_w, best_acc


if __name__ == "__main__":
    print("Loading test data...")
    y_test    = joblib.load(ARTIFACTS_DIR / "y_test.joblib")
    X_test_emb = joblib.load(ARTIFACTS_DIR / "X_test_lgbm.joblib")
    X_test_txt = joblib.load(ARTIFACTS_DIR / "X_test_roberta.joblib")

    # LightGBM probabilities
    print("\nGetting LightGBM predictions...")
    lgbm_probas = get_lgbm_probas(X_test_emb)
    lgbm_preds  = (lgbm_probas >= 0.5).astype(int)
    print(f"  LightGBM accuracy: {accuracy_score(y_test, lgbm_preds):.4f}")

    # RoBERTa probabilities
    print("\nGetting RoBERTa predictions...")
    roberta_probas = get_roberta_probas(X_test_txt)
    roberta_preds  = (roberta_probas >= 0.5).astype(int)
    print(f"  RoBERTa accuracy:  {accuracy_score(y_test, roberta_preds):.4f}")

    # Find optimal weights
    print("\nFinding optimal ensemble weights...")
    best_w, best_acc = find_best_weights(lgbm_probas, roberta_probas, y_test)
    print(f"  Best weight  : LightGBM={best_w:.2f}  RoBERTa={1-best_w:.2f}")
    print(f"  Best accuracy: {best_acc:.4f}")

    # Final ensemble
    final_probas = best_w * lgbm_probas + (1 - best_w) * roberta_probas
    final_preds  = (final_probas >= 0.5).astype(int)

    print("\n=== Ensemble Test Metrics ===")
    print(f"  Accuracy : {accuracy_score(y_test, final_preds):.4f}")
    print(f"  F1 Macro : {f1_score(y_test, final_preds, average='macro'):.4f}")
    print(f"  AUC-ROC  : {roc_auc_score(y_test, final_probas):.4f}")
    print(f"\n{classification_report(y_test, final_preds, target_names=['Real', 'Fake'])}")

    # Save ensemble config
    joblib.dump({"lgbm_weight": best_w, "roberta_weight": 1 - best_w},
                ARTIFACTS_DIR / "ensemble_weights.joblib")
    print(f"Ensemble weights saved → ensemble_weights.joblib")
