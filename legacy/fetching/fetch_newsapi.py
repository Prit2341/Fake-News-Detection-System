"""
Fetch news articles from NewsAPI (India + Global top headlines).
Stores results in PostgreSQL with label = 0 (real, from trusted API source).
Uses only valid NewsAPI categories to avoid wasting requests.
Skips already-fetched URLs automatically.
"""

import os
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/top-headlines"

# Valid NewsAPI categories + India country code
FETCHES = [
    {"country": "in", "category": "general"},
    {"country": "in", "category": "business"},
    {"country": "in", "category": "technology"},
    {"country": "in", "category": "health"},
    {"country": "in", "category": "science"},
    {"country": "us", "category": "general"},
    {"country": "us", "category": "technology"},
    {"country": "us", "category": "health"},
]

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "fakenews"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )

def upsert_source(cur, name, url):
    cur.execute("""
        INSERT INTO sources (name, url, type, trust_level)
        VALUES (%s, %s, 'api', 'real')
        ON CONFLICT (url) DO NOTHING
        RETURNING id
    """, (name, url))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("SELECT id FROM sources WHERE url = %s", (url,))
    return cur.fetchone()[0]

def fetch_headlines(country, category):
    params = {
        "country":  country,
        "category": category,
        "pageSize": 100,
        "apiKey":   API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise ValueError(data.get("message", "Unknown API error"))
    return data.get("articles", [])

def store_articles(cur, articles, source_id):
    rows = []
    for a in articles:
        url = a.get("url")
        if not url or url == "https://removed.com":
            continue
        title   = (a.get("title") or "")[:500]
        content = a.get("content") or a.get("description") or ""
        pub_at  = a.get("publishedAt")
        if pub_at:
            try:
                pub_at = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
            except Exception:
                pub_at = None
        rows.append((title, content, url, source_id, 0, None, pub_at))

    if not rows:
        return 0

    execute_values(cur, """
        INSERT INTO articles (title, content, url, source_id, label, subject, published_at)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
    """, rows)
    return len(rows)

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: NEWS_API_KEY not set in .env")
        exit(1)

    conn = get_connection()
    cur  = conn.cursor()
    total = 0

    for fetch in FETCHES:
        country  = fetch["country"]
        category = fetch["category"]
        name     = f"NewsAPI [{country.upper()}] {category}"
        src_url  = f"newsapi://top-headlines/{country}/{category}"

        source_id = upsert_source(cur, name, src_url)
        try:
            articles = fetch_headlines(country, category)
            count    = store_articles(cur, articles, source_id)
            cur.execute("""
                INSERT INTO fetch_log (source_id, article_count, status)
                VALUES (%s, %s, 'success')
            """, (source_id, count))
            print(f"  [{name}] {count} articles stored")
            total += count
        except Exception as e:
            cur.execute("""
                INSERT INTO fetch_log (source_id, article_count, status)
                VALUES (%s, 0, 'failed')
            """, (source_id,))
            print(f"  [{name}] FAILED: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. Total stored: {total} articles")
