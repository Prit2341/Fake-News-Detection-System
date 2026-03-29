"""
Fine-tune XLM-RoBERTa for fake news binary classification.
Uses mixed precision (fp16) to fit within GTX 1650 4GB VRAM.
"""

import os
import joblib
import numpy as np
import pandas as pd
import psycopg2
import torch
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

SEED          = 42
MODEL_NAME    = "xlm-roberta-base"
MAX_LEN       = 256        # keep low for 4GB VRAM
BATCH_SIZE    = 8          # small batch for 4GB VRAM
EPOCHS        = 3
LR            = 2e-5
ARTIFACTS_DIR = Path("d:/Fake_News_Detection/artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
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


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "fakenews"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


if __name__ == "__main__":
    # Load data
    print("Loading articles from PostgreSQL...")
    conn = get_connection()
    df   = pd.read_sql("SELECT title, content, label FROM articles WHERE label IS NOT NULL", conn)
    conn.close()
    print(f"  Loaded {len(df):,} articles")

    # Balance
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]
    min_count = min(len(real), len(fake))
    df = pd.concat([
        real.sample(n=min_count, random_state=SEED),
        fake.sample(n=min_count, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    df["title"]   = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"]    = (df["title"] + " " + df["content"]).str.strip()
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    print(f"  After balancing & filtering: {len(df):,}")

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp)
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # Tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset   = NewsDataset(X_val,   y_val,   tokenizer)
    test_dataset  = NewsDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    best_val_acc  = 0
    best_val_loss = float("inf")

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss    = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            if step % 100 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Val   Acc  : {val_acc:.4f}")
        print(f"  Val   F1   : {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(ARTIFACTS_DIR / "roberta_model")
            tokenizer.save_pretrained(ARTIFACTS_DIR / "roberta_model")
            print(f"  Best model saved (val_acc={best_val_acc:.4f})")

    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    model_best = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS_DIR / "roberta_model")
    model_best = model_best.to(device)
    test_loss, test_acc, test_f1 = evaluate(model_best, test_loader)
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1       : {test_f1:.4f}")

    # Save test set for ensemble
    joblib.dump(X_test, ARTIFACTS_DIR / "X_test_roberta.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")

    print("\nRoBERTa training done.")
