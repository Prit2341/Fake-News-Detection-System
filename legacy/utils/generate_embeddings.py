"""
Generate sentence embeddings for all articles using
paraphrase-multilingual-MiniLM-L12-v2 (supports Hindi + English).
Saves embeddings + labels to artifacts for LightGBM training.
"""

import os
import joblib
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

SEED        = 42
BATCH_SIZE  = 256
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
ARTIFACTS_DIR = Path("d:/Fake_News_Detection/artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "fakenews"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )


if __name__ == "__main__":
    # Load from PostgreSQL
    print("Loading articles from PostgreSQL...")
    conn = get_connection()
    df = pd.read_sql(
        "SELECT title, content, label FROM articles WHERE label IS NOT NULL", conn
    )
    conn.close()
    print(f"  Loaded {len(df):,} articles")

    # Balance classes
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]
    min_count = min(len(real), len(fake))
    df = pd.concat([
        real.sample(n=min_count, random_state=SEED),
        fake.sample(n=min_count, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"  After balancing: {len(df):,} articles ({min_count:,} each class)")

    # Combine title + content
    df["title"]   = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"]    = (df["title"] + " " + df["content"]).str.strip()
    df = df[df["text"].str.len() > 50].reset_index(drop=True)
    print(f"  After filtering short texts: {len(df):,}")

    # Load embedding model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print(f"  Model loaded on GPU")

    # Generate embeddings
    print(f"\nGenerating embeddings (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    y = df["label"].values

    # Save
    joblib.dump(embeddings, ARTIFACTS_DIR / "embeddings.joblib")
    joblib.dump(y,          ARTIFACTS_DIR / "y_labels.joblib")
    print(f"\nSaved: embeddings.joblib ({embeddings.shape}), y_labels.joblib")
    print("Embedding generation done.")
