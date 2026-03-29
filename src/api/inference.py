"""
Inference engine — loads models once, serves predictions.
LightGBM + XLM-RoBERTa ensemble.
"""

import joblib, torch, numpy as np
from pathlib import Path
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN   = 256


# ── Lazy singletons ───────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_smodel():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device=str(DEVICE)
    )

@lru_cache(maxsize=1)
def _load_lgbm():
    return joblib.load(ARTIFACTS / "lgbm_model.joblib")

@lru_cache(maxsize=1)
def _load_roberta():
    tok   = AutoTokenizer.from_pretrained(ARTIFACTS / "roberta_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        ARTIFACTS / "roberta_model"
    ).to(DEVICE)
    model.eval()
    return tok, model

@lru_cache(maxsize=1)
def _load_weights():
    w = joblib.load(ARTIFACTS / "ensemble_weights.joblib")
    return float(w["lgbm_weight"]), float(w["roberta_weight"])


# ── Prediction ────────────────────────────────────────────────

def predict(text: str) -> dict:
    """
    Returns per-model confidence + ensemble result for a single text.
    """
    # 1. Embedding
    smodel = _load_smodel()
    emb    = smodel.encode([text], normalize_embeddings=True)

    # 2. LightGBM
    lgbm       = _load_lgbm()
    lgbm_prob  = float(lgbm.predict_proba(emb)[0, 1])
    lgbm_label = int(lgbm_prob >= 0.5)

    # 3. RoBERTa
    tok, rob = _load_roberta()
    enc = tok(
        text, truncation=True, padding=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = rob(**enc)
        else:
            out = rob(**enc)
    rob_probs  = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
    rob_prob   = float(rob_probs[1])
    rob_label  = int(rob_prob >= 0.5)

    # 4. Ensemble
    lw, rw        = _load_weights()
    ens_prob      = lw * lgbm_prob + rw * rob_prob
    ens_label     = int(ens_prob >= 0.5)
    ens_confidence = float(max(ens_prob, 1 - ens_prob))

    return {
        "text_snippet": text[:120] + "..." if len(text) > 120 else text,
        "lgbm": {
            "label":      "FAKE" if lgbm_label else "REAL",
            "confidence": round(max(lgbm_prob, 1 - lgbm_prob) * 100, 2),
            "fake_prob":  round(lgbm_prob * 100, 2),
            "real_prob":  round((1 - lgbm_prob) * 100, 2),
        },
        "roberta": {
            "label":      "FAKE" if rob_label else "REAL",
            "confidence": round(max(rob_prob, 1 - rob_prob) * 100, 2),
            "fake_prob":  round(rob_prob * 100, 2),
            "real_prob":  round((1 - rob_prob) * 100, 2),
        },
        "ensemble": {
            "label":      "FAKE" if ens_label else "REAL",
            "confidence": round(ens_confidence * 100, 2),
            "fake_prob":  round(ens_prob * 100, 2),
            "real_prob":  round((1 - ens_prob) * 100, 2),
            "lgbm_weight": round(lw, 2),
            "roberta_weight": round(rw, 2),
        },
        "device": str(DEVICE),
    }


def get_metrics() -> dict:
    m = joblib.load(ARTIFACTS / "metrics.joblib")
    return {k: round(float(v) * 100, 4) for k, v in m.items()}
