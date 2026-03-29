"""
Fetch fake/debunked news from fact-checker feeds (India + Global).
Stores results in PostgreSQL articles table with label = 1 (fake).
"""

import os
import feedparser
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from email.utils import parsedate_to_datetime

load_dotenv()

# Fact-checker feeds → label FAKE
FACTCHECK_SOURCES = [
    # India
    ("Alt News",              "https://www.altnews.in/feed/"),
    ("BOOM Live",             "https://www.boomlive.in/feed"),
    ("Vishvas News",          "https://www.vishvasnews.com/feed/"),
    ("FactChecker.in",        "https://www.factchecker.in/feed/"),
    ("India Today Fact Check","https://www.indiatoday.in/fact-check/rss"),
    # Global
    ("Snopes",                "https://www.snopes.com/feed/"),
    ("PolitiFact",            "https://www.politifact.com/rss/rulings/pants-fire/feed/"),
    ("AFP Fact Check",        "https://factcheck.afp.com/feed"),
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
        VALUES (%s, %s, 'rss', 'fake')
        ON CONFLICT (url) DO NOTHING
        RETURNING id
    """, (name, url))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("SELECT id FROM sources WHERE url = %s", (url,))
    return cur.fetchone()[0]

def parse_date(entry):
    for field in ("published", "updated"):
        val = entry.get(field)
        if val:
            try:
                return parsedate_to_datetime(val)
            except Exception:
                pass
    return None

def fetch_and_store(cur, name, feed_url, source_id):
    feed = feedparser.parse(feed_url)
    entries = feed.get("entries", [])

    rows = []
    for entry in entries:
        url = entry.get("link") or entry.get("id")
        if not url:
            continue
        title   = entry.get("title", "")[:500]
        content = entry.get("summary") or entry.get("content", [{}])[0].get("value", "")
        pub_at  = parse_date(entry)

        rows.append((title, content, url, source_id, 1, None, pub_at))

    if not rows:
        return 0

    execute_values(cur, """
        INSERT INTO articles (title, content, url, source_id, label, subject, published_at)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
    """, rows)

    return len(rows)

if __name__ == "__main__":
    conn = get_connection()
    cur = conn.cursor()

    total = 0
    for name, url in FACTCHECK_SOURCES:
        source_id = upsert_source(cur, name, url)
        try:
            count = fetch_and_store(cur, name, url, source_id)
            cur.execute("""
                INSERT INTO fetch_log (source_id, article_count, status)
                VALUES (%s, %s, 'success')
            """, (source_id, count))
            print(f"  [{name}] {count} articles fetched")
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

    print(f"\nDone. Total fetched: {total} fake/debunked articles")
