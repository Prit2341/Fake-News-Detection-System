"""
Create all PostgreSQL tables for the Fake News Detection pipeline.
Run this once before anything else.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "fakenews"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    type        VARCHAR(50),        -- 'rss', 'csv', 'api'
    trust_level VARCHAR(10),        -- 'real', 'fake'
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS articles (
    id           SERIAL PRIMARY KEY,
    title        TEXT,
    content      TEXT,
    url          TEXT UNIQUE,
    source_id    INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    label        SMALLINT,          -- 0 = real, 1 = fake
    subject      VARCHAR(255),
    published_at TIMESTAMP,
    fetched_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id            SERIAL PRIMARY KEY,
    source_id     INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    fetched_at    TIMESTAMP DEFAULT NOW(),
    article_count INTEGER,
    status        VARCHAR(50)        -- 'success', 'failed'
);

CREATE INDEX IF NOT EXISTS idx_articles_label     ON articles(label);
CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_fetched   ON articles(fetched_at);
"""

def create_database_if_not_exists():
    """Connect to default 'postgres' db and create 'fakenews' if missing."""
    db_name = os.getenv("DB_NAME", "fakenews")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname="postgres",
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{db_name}"')
        print(f"Database '{db_name}' created.")
    else:
        print(f"Database '{db_name}' already exists.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    create_database_if_not_exists()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(SCHEMA)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created: sources, articles, fetch_log")
