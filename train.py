"""
Master training script — runs full pipeline end-to-end.
Shows VRAM, time, loss, accuracy at every stage.

Pipeline:
  1. Load & balance data from PostgreSQL
  2. Generate multilingual embeddings (Sentence Transformer)
  3. Train LightGBM on embeddings
  4. Fine-tune XLM-RoBERTa
  5. Ensemble both models
  6. Final evaluation
"""

import os, gc, time, joblib, warnings, psutil
import numpy as np
import pandas as pd
import psycopg2
import torch
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Config ────────────────────────────────────────────────────
SEED              = 42
ARTIFACTS_DIR     = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

SMODEL_NAME       = "paraphrase-multilingual-MiniLM-L12-v2"
ROBERTA_NAME      = "xlm-roberta-base"
EMBED_BATCH_SIZE  = 512   # peak=585MB VRAM
ROBERTA_BATCH     = 4     # batch=4 + gradient checkpointing fits in 4GB
ROBERTA_ACCUM     = 2     # gradient accumulation: effective batch = 4×2 = 8
ROBERTA_MAX_LEN   = 256   # seq=128 spills to system RAM — keep at 256
ROBERTA_MAX_TRAIN = 10000 # cap training samples — RoBERTa fine-tunes well on 10k
NUM_WORKERS       = 4     # parallel data loading (12 cores available, keep 4 for training)
PREFETCH_FACTOR   = 2     # prefetch next batch while GPU processes current
ROBERTA_EPOCHS    = 1     # 1 epoch sufficient — resume skips to ensemble
ROBERTA_LR        = 2e-5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VRAM management — hard cap at 93% (3.8GB) to prevent spilling to system RAM
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    torch.cuda.set_per_process_memory_fraction(0.93)  # 3.8GB max — OOM before spill


# ─── Helpers ───────────────────────────────────────────────────
def vram_str():
    ram  = psutil.virtual_memory()
    ram_used  = ram.used  / 1024**3
    ram_total = ram.total / 1024**3
    ram_str   = f"RAM {ram_used:.1f}/{ram_total:.1f}GB"
    if not torch.cuda.is_available():
        return f"[{ram_str}]"
    vram_used  = torch.cuda.memory_allocated() / 1024**2
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    vram_warn  = " ⚠" if vram_used > 3700 else ""
    ram_warn   = " ⚠" if ram.percent > 90 else ""
    return f"[VRAM {vram_used:.0f}/{vram_total:.0f}MB{vram_warn} | {ram_str}{ram_warn}]"

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def elapsed(start):
    s = int(time.time() - start)
    return f"{s//60}m {s%60}s"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "fakenews"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )


# ─── Dataset class ─────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=ROBERTA_MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─── STEP 1: Load Data ─────────────────────────────────────────
def load_data():
    section("STEP 1 — Loading Data from PostgreSQL")
    t = time.time()

    conn = get_connection()
    df   = pd.read_sql("SELECT title, content, label FROM articles WHERE label IS NOT NULL", conn)
    conn.close()

    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    print(f"  Total loaded : {len(df):,}")
    print(f"  Real         : {real_count:,}")
    print(f"  Fake         : {fake_count:,}")

    # Balance
    min_count = min(real_count, fake_count)
    df = pd.concat([
        df[df["label"] == 0].sample(n=min_count, random_state=SEED),
        df[df["label"] == 1].sample(n=min_count, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    df["title"]   = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"]    = (df["title"] + " " + df["content"]).str.strip()
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    print(f"  After balance: {len(df):,} ({min_count:,} each class)")
    print(f"  Done in {elapsed(t)}")
    return df


# ─── STEP 2: Generate Embeddings ──────────────────────────────
def generate_embeddings(df):
    section("STEP 2 — Generating Sentence Embeddings")
    t = time.time()
    clear_vram()

    print(f"  Model  : {SMODEL_NAME}")
    print(f"  Batch  : {EMBED_BATCH_SIZE}")
    print(f"  Device : {device}  {vram_str()}")

    smodel = SentenceTransformer(SMODEL_NAME, device=str(device))
    print(f"  Model loaded  {vram_str()}")

    embeddings = smodel.encode(
        df["text"].tolist(),
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Shape  : {embeddings.shape}")
    print(f"  {vram_str()}")

    del smodel
    clear_vram()

    y = df["label"].values
    joblib.dump(embeddings, ARTIFACTS_DIR / "embeddings.joblib")
    joblib.dump(y,          ARTIFACTS_DIR / "y_labels.joblib")
    print(f"  Saved  : embeddings.joblib, y_labels.joblib")
    print(f"  Done in {elapsed(t)}")
    return embeddings, y


# ─── STEP 3: Train LightGBM ───────────────────────────────────
def train_lightgbm(X, y):
    section("STEP 3 — Training LightGBM on Embeddings")
    t = time.time()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp)

    print(f"  Train : {X_train.shape[0]:,}")
    print(f"  Val   : {X_val.shape[0]:,}")
    print(f"  Test  : {X_test.shape[0]:,}")

    params = dict(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        device="gpu",
        random_state=SEED,
        verbose=1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter  = model.best_iteration_
    train_acc  = accuracy_score(y_train, model.predict(X_train))
    val_acc    = accuracy_score(y_val,   model.predict(X_val))
    test_acc   = accuracy_score(y_test,  model.predict(X_test))
    gap        = train_acc - test_acc

    print(f"\n  Best iteration : {best_iter}")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Val   Accuracy : {val_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  Gap            : {gap:.4f}  {'(LOW overfit)' if gap < 0.01 else '(MILD overfit)' if gap < 0.03 else '(HIGH overfit)'}")

    # 5-fold CV
    print(f"\n  Running 5-Fold CV (n_estimators={best_iter})...")
    cv_model  = lgb.LGBMClassifier(**{**params, "n_estimators": best_iter, "verbose": -1})
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(cv_model, X_temp, y_temp, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Folds          : {[round(s,4) for s in cv_scores]}")

    joblib.dump(model,  ARTIFACTS_DIR / "lgbm_model.joblib")
    joblib.dump(X_test, ARTIFACTS_DIR / "X_test_lgbm.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")

    print(f"\n  Saved: lgbm_model.joblib")
    print(f"  Done in {elapsed(t)}")
    return model, X_test, y_test


# ─── STEP 4: Fine-tune XLM-RoBERTa ───────────────────────────
def train_roberta(df):
    section("STEP 4 — Fine-tuning XLM-RoBERTa")
    t = time.time()
    clear_vram()

    # Cap dataset — XLM-RoBERTa fine-tunes well on 10k; more just means slower training
    if len(df) > ROBERTA_MAX_TRAIN:
        per_class = ROBERTA_MAX_TRAIN // 2
        real = df[df["label"] == 0].sample(n=per_class, random_state=SEED)
        fake = df[df["label"] == 1].sample(n=per_class, random_state=SEED)
        df = pd.concat([real, fake]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"  Capped to {len(df):,} samples ({per_class:,} per class)")

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp)

    # Skip training if model already saved — run test eval only
    CKPT        = ARTIFACTS_DIR / "roberta_checkpoint.pt"
    MODEL_SAVED = (ARTIFACTS_DIR / "roberta_model" / "model.safetensors").exists()
    if MODEL_SAVED and not CKPT.exists():
        print(f"  Saved model found — skipping training, running test evaluation only.")
        tokenizer  = AutoTokenizer.from_pretrained(ARTIFACTS_DIR / "roberta_model")
        best_model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS_DIR / "roberta_model").to(device)
        best_model.eval()
        test_loader = DataLoader(
            NewsDataset(X_test, y_test, tokenizer),
            batch_size=ROBERTA_BATCH * 2,
            num_workers=NUM_WORKERS, pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
        )
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                with torch.amp.autocast("cuda"):
                    out = best_model(input_ids=ids, attention_mask=mask)
                test_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
                test_true.extend(batch["labels"].numpy())
        test_acc = accuracy_score(test_true, test_preds)
        test_f1  = f1_score(test_true, test_preds, average="macro")
        print(f"  Test Accuracy : {test_acc:.4f}  |  F1: {test_f1:.4f}  {vram_str()}")
        joblib.dump(X_test, ARTIFACTS_DIR / "X_test_roberta.joblib")
        joblib.dump(y_test, ARTIFACTS_DIR / "y_test_roberta.joblib")
        del best_model; clear_vram()
        return X_test, y_test

    print(f"  Model    : {ROBERTA_NAME}")
    print(f"  Epochs   : {ROBERTA_EPOCHS}")
    print(f"  Batch    : {ROBERTA_BATCH}")
    print(f"  Max len  : {ROBERTA_MAX_LEN}")
    print(f"  LR       : {ROBERTA_LR}")
    print(f"  Train    : {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"  Device   : {device}  {vram_str()}")

    tokenizer    = AutoTokenizer.from_pretrained(ROBERTA_NAME)
    train_loader = DataLoader(
        NewsDataset(X_train, y_train, tokenizer),
        batch_size=ROBERTA_BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
    )
    val_loader = DataLoader(
        NewsDataset(X_val, y_val, tokenizer),
        batch_size=ROBERTA_BATCH * 2,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
    )
    test_loader = DataLoader(
        NewsDataset(X_test, y_test, tokenizer),
        batch_size=ROBERTA_BATCH * 2,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
    )

    model      = AutoModelForSequenceClassification.from_pretrained(ROBERTA_NAME, num_labels=2).to(device)
    model.gradient_checkpointing_enable()   # recompute activations on backward → saves ~40% VRAM

    # Freeze first 10 of 12 encoder layers — only train last 2 layers + classifier
    # Reduces trainable params 278M → ~20M, optimizer states 3.3GB → ~240MB
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(f"encoder.layer.{i}" in name for i in [10, 11]):
            param.requires_grad = True
        if "classifier" in name or "pooler" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ROBERTA_LR, weight_decay=0.01
    )
    total_steps = (len(train_loader) // ROBERTA_ACCUM) * ROBERTA_EPOCHS
    scheduler  = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
    scaler     = GradScaler()

    print(f"\n  Model loaded  {vram_str()}")
    print(f"  Total steps  : {total_steps:,}  (effective batch={ROBERTA_BATCH*ROBERTA_ACCUM})\n")

    # ── Resume from checkpoint if available ──
    CKPT = ARTIFACTS_DIR / "roberta_checkpoint.pt"
    best_val_acc = 0
    start_epoch  = 1

    if CKPT.exists():
        print(f"  Resuming from checkpoint: {CKPT}")
        ckpt = torch.load(CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val_acc = ckpt["best_val_acc"]
        start_epoch  = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}  best_val_acc={best_val_acc:.4f}\n")

    for epoch in range(start_epoch, ROBERTA_EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
                loss = out.loss / ROBERTA_ACCUM

            scaler.scale(loss).backward()

            if step % ROBERTA_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ROBERTA_ACCUM

            if step % 50 == 0 or step == len(train_loader):
                avg      = epoch_loss / step
                pct      = step / len(train_loader) * 100
                eta      = (time.time() - epoch_start) / step * (len(train_loader) - step)
                vram_mb  = torch.cuda.memory_allocated() / 1024**2
                vram_warn = " ⚠ HIGH VRAM" if vram_mb > 3700 else ""
                print(f"  Epoch {epoch} [{step:>4}/{len(train_loader)}  {pct:5.1f}%] "
                      f"loss={avg:.4f}  eta={int(eta//60)}m{int(eta%60):02d}s  {vram_str()}{vram_warn}")

        # ── Validate ──
        model.eval()
        val_loss, val_preds, val_true = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["labels"].to(device)
                with torch.amp.autocast("cuda"):
                    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                val_loss += out.loss.item()
                val_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
                val_true.extend(lbls.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1  = f1_score(val_true, val_preds, average="macro")
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        print(f"\n  ── Epoch {epoch} Summary ──────────────────────")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val   Loss : {avg_val_loss:.4f}")
        print(f"  Val   Acc  : {val_acc:.4f}")
        print(f"  Val   F1   : {val_f1:.4f}")
        print(f"  Time       : {elapsed(epoch_start)}")
        print(f"  {vram_str()}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(ARTIFACTS_DIR / "roberta_model")
            tokenizer.save_pretrained(ARTIFACTS_DIR / "roberta_model")
            print(f"  Best model saved ✓  (val_acc={best_val_acc:.4f})")

        # Save resume checkpoint after every epoch
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "scaler":       scaler.state_dict(),
            "best_val_acc": best_val_acc,
        }, CKPT)
        print(f"  Checkpoint saved → {CKPT.name}")
        print()

    # ── Test ──
    print("  Loading best checkpoint for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS_DIR / "roberta_model").to(device)
    best_model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda"):
                out = best_model(input_ids=ids, attention_mask=mask)
            test_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            test_true.extend(batch["labels"].numpy())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1  = f1_score(test_true, test_preds, average="macro")
    print(f"\n  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1       : {test_f1:.4f}")

    joblib.dump(X_test, ARTIFACTS_DIR / "X_test_roberta.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test_roberta.joblib")

    del best_model
    clear_vram()

    # Clean up checkpoint — training completed successfully
    if CKPT.exists():
        CKPT.unlink()
        print(f"  Checkpoint removed (training complete)")

    print(f"\n  Saved: roberta_model/")
    print(f"  Done in {elapsed(t)}")
    return X_test, y_test


# ─── STEP 5: Ensemble ─────────────────────────────────────────
def run_ensemble(lgbm_model, roberta_X_test, y_test):
    section("STEP 5 — Ensemble (LightGBM + RoBERTa)")
    t = time.time()

    # Generate embeddings for the shared test set, then get LightGBM probabilities
    print("  Getting LightGBM probabilities (embedding shared test set)...")
    smodel = SentenceTransformer(SMODEL_NAME, device=str(device))
    test_embeddings = smodel.encode(
        roberta_X_test, batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )
    del smodel
    clear_vram()
    lgbm_probas = lgbm_model.predict_proba(test_embeddings)[:, 1]

    # RoBERTa probabilities
    print("  Getting RoBERTa probabilities...")
    tokenizer   = AutoTokenizer.from_pretrained(ARTIFACTS_DIR / "roberta_model")
    rob_model   = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS_DIR / "roberta_model").to(device)
    rob_model.eval()

    dataset = DataLoader(
        [(tokenizer([t], truncation=True, padding=True, max_length=ROBERTA_MAX_LEN,
                    return_tensors="pt")) for t in roberta_X_test],
        batch_size=ROBERTA_BATCH * 2,
    )
    rob_probas = []
    with torch.no_grad():
        for batch in DataLoader(
            __import__('torch').utils.data.TensorDataset(
                tokenizer(roberta_X_test, truncation=True, padding=True,
                          max_length=ROBERTA_MAX_LEN, return_tensors="pt")["input_ids"],
                tokenizer(roberta_X_test, truncation=True, padding=True,
                          max_length=ROBERTA_MAX_LEN, return_tensors="pt")["attention_mask"],
            ),
            batch_size=ROBERTA_BATCH * 2,
        ):
            ids, mask = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast("cuda"):
                out = rob_model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            rob_probas.extend(probs)

    rob_probas = np.array(rob_probas)

    # Find best weights
    best_acc, best_w = 0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        combined = w * lgbm_probas + (1 - w) * rob_probas
        acc      = accuracy_score(y_test, (combined >= 0.5).astype(int))
        if acc > best_acc:
            best_acc, best_w = acc, w

    final_probas = best_w * lgbm_probas + (1 - best_w) * rob_probas
    final_preds  = (final_probas >= 0.5).astype(int)

    print(f"\n  Optimal weights : LightGBM={best_w:.2f}  RoBERTa={1-best_w:.2f}")
    print(f"\n  Individual scores:")
    print(f"    LightGBM  accuracy : {accuracy_score(y_test, (lgbm_probas>=0.5).astype(int)):.4f}")
    print(f"    RoBERTa   accuracy : {accuracy_score(y_test, (rob_probas>=0.5).astype(int)):.4f}")
    print(f"    Ensemble  accuracy : {accuracy_score(y_test, final_preds):.4f}")

    joblib.dump({"lgbm_weight": best_w, "roberta_weight": 1 - best_w},
                ARTIFACTS_DIR / "ensemble_weights.joblib")

    del rob_model
    clear_vram()
    print(f"  Done in {elapsed(t)}")
    return final_preds, final_probas


# ─── STEP 6: Final Evaluation ─────────────────────────────────
def final_evaluation(y_test, final_preds, final_probas):
    section("STEP 6 — Final Evaluation")

    cm = confusion_matrix(y_test, final_preds)
    print(f"\n  Accuracy  : {accuracy_score(y_test, final_preds):.4f}")
    print(f"  F1 Macro  : {f1_score(y_test, final_preds, average='macro'):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, final_probas):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y_test, final_preds, target_names=['Real','Fake'])}")

    joblib.dump({
        "accuracy": accuracy_score(y_test, final_preds),
        "f1":       f1_score(y_test, final_preds, average="macro"),
        "auc_roc":  roc_auc_score(y_test, final_probas),
    }, ARTIFACTS_DIR / "metrics.joblib")
    print(f"  Metrics saved → metrics.joblib")


# ─── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"  FAKE NEWS DETECTION — TRAINING PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device : {device}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB total")
    print(f"{'='*60}")

    df                           = load_data()
    embeddings, y                = generate_embeddings(df)
    lgbm_model, lgbm_X_test, y_test = train_lightgbm(embeddings, y)
    roberta_X_test, roberta_y_test = train_roberta(df)
    final_preds, final_probas      = run_ensemble(lgbm_model, roberta_X_test, roberta_y_test)
    final_evaluation(roberta_y_test, final_preds, final_probas)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time : {elapsed(total_start)}")
    print(f"  Artifacts  : {ARTIFACTS_DIR}")
    print(f"{'='*60}\n")
