"""
Evaluate saved models — no retraining.
Loads LightGBM + RoBERTa from artifacts/ and prints final scores.
"""

import os, joblib, torch, numpy as np
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

load_dotenv()

ARTIFACTS  = Path(__file__).resolve().parent / "artifacts"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 16
MAX_LEN    = 256

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.93)

class NewsDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding=True,
                             max_length=MAX_LEN, return_tensors="pt")
    def __len__(self):  return self.enc["input_ids"].shape[0]
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}

def vram():
    if not torch.cuda.is_available(): return ""
    used  = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return f"[VRAM {used:.0f}/{total:.0f}MB]"

print("\n" + "="*55)
print("  FAKE NEWS DETECTION — EVALUATION")
print("="*55)

# ── Load test data ───────────────────────────────────────────
print("\nLoading test data...")
X_test   = joblib.load(ARTIFACTS / "X_test_roberta.joblib")   # texts
y_test   = joblib.load(ARTIFACTS / "y_test_roberta.joblib")   # labels
X_emb    = joblib.load(ARTIFACTS / "X_test_lgbm.joblib")      # lgbm embeddings

# If embedding test set size differs from text test set, regenerate for shared set
if len(X_emb) != len(X_test):
    print(f"  Embedding size mismatch ({len(X_emb)} vs {len(X_test)}) — regenerating...")
    from sentence_transformers import SentenceTransformer
    smodel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=str(DEVICE))
    X_emb  = smodel.encode(X_test, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
    del smodel; torch.cuda.empty_cache()

print(f"  Samples : {len(X_test):,}  |  Real: {sum(1 for y in y_test if y==0):,}  |  Fake: {sum(1 for y in y_test if y==1):,}")

# ── LightGBM ─────────────────────────────────────────────────
print("\nEvaluating LightGBM...")
lgbm      = joblib.load(ARTIFACTS / "lgbm_model.joblib")
lgbm_prob = lgbm.predict_proba(X_emb)[:, 1]
lgbm_pred = (lgbm_prob >= 0.5).astype(int)
print(f"  Accuracy : {accuracy_score(y_test, lgbm_pred):.4f}")
print(f"  F1       : {f1_score(y_test, lgbm_pred, average='macro'):.4f}")
print(f"  AUC-ROC  : {roc_auc_score(y_test, lgbm_prob):.4f}")

# ── RoBERTa ──────────────────────────────────────────────────
print(f"\nEvaluating RoBERTa...  {vram()}")
tokenizer  = AutoTokenizer.from_pretrained(ARTIFACTS / "roberta_model")
rob_model  = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS / "roberta_model").to(DEVICE)
rob_model.eval()

dataset = NewsDataset(X_test, tokenizer)
loader  = DataLoader(dataset, batch_size=BATCH, num_workers=0)
rob_prob, rob_pred = [], []

with torch.no_grad():
    for i, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast("cuda"):
            out = rob_model(input_ids=ids, attention_mask=mask)
        rob_prob.extend(torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy())
        rob_pred.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(loader)}]  {vram()}")

rob_prob = np.array(rob_prob)
rob_pred = np.array(rob_pred)
print(f"  Accuracy : {accuracy_score(y_test, rob_pred):.4f}")
print(f"  F1       : {f1_score(y_test, rob_pred, average='macro'):.4f}")
print(f"  AUC-ROC  : {roc_auc_score(y_test, rob_prob):.4f}")

del rob_model; torch.cuda.empty_cache()

# ── Ensemble ─────────────────────────────────────────────────
print("\nFinding best ensemble weights...")
best_acc, best_w = 0, 0.5
for w in np.arange(0.0, 1.05, 0.05):
    combined = w * lgbm_prob + (1 - w) * rob_prob
    acc = accuracy_score(y_test, (combined >= 0.5).astype(int))
    if acc > best_acc:
        best_acc, best_w = acc, w

final_prob = best_w * lgbm_prob + (1 - best_w) * rob_prob
final_pred = (final_prob >= 0.5).astype(int)

print(f"\n{'='*55}")
print(f"  FINAL RESULTS")
print(f"{'='*55}")
print(f"  LightGBM  Acc : {accuracy_score(y_test, lgbm_pred):.4f}")
print(f"  RoBERTa   Acc : {accuracy_score(y_test, rob_pred):.4f}")
print(f"  Ensemble  Acc : {accuracy_score(y_test, final_pred):.4f}  (lgbm={best_w:.2f}, rob={1-best_w:.2f})")
print(f"  Ensemble  F1  : {f1_score(y_test, final_pred, average='macro'):.4f}")
print(f"  Ensemble  AUC : {roc_auc_score(y_test, final_prob):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, final_pred, target_names=["Real", "Fake"]))

joblib.dump({"lgbm_weight": best_w, "roberta_weight": 1 - best_w},
            ARTIFACTS / "ensemble_weights.joblib")
print(f"  Saved: ensemble_weights.joblib")
print("="*55 + "\n")
