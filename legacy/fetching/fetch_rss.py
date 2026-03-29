"""
Fetch real news articles from trusted RSS feeds (India + Global).
Stores results in PostgreSQL articles table with label = 0 (real).
Skips already-fetched URLs to preserve daily API quota.
"""

import os
import feedparser
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from dotenv import load_dotenv
from email.utils import parsedate_to_datetime

load_dotenv()

# India + Global trusted RSS feeds → label REAL
RSS_SOURCES = [
    # India
    ("Times of India - Top Stories",  "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("Times of India - India",        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"),
    ("NDTV - Top Stories",            "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("India Today",                   "https://www.indiatoday.in/rss/home"),
    ("Hindustan Times",               "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"),
    # Global
    ("BBC World",                     "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Al Jazeera",                    "https://www.aljazeera.com/xml/rss/all.xml"),
    ("NPR News",                      "https://feeds.npr.org/1001/rss.xml"),
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
        VALUES (%s, %s, 'rss', 'real')
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

        rows.append((title, content, url, source_id, 0, None, pub_at))

    if not rows:
        return 0

    execute_values(cur, """
        INSERT INTO articles (title, content, url, source_id, label, subject, published_at)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
    """, rows)

    # Count how many were actually new
    return len(rows)

if __name__ == "__main__":
    conn = get_connection()
    cur = conn.cursor()

    total = 0
    for name, url in RSS_SOURCES:
        source_id = upsert_source(cur, name, url)
        count = fetch_and_store(cur, name, url, source_id)

        # Log the fetch
        cur.execute("""
            INSERT INTO fetch_log (source_id, article_count, status)
            VALUES (%s, %s, 'success')
        """, (source_id, count))

        print(f"  [{name}] {count} articles fetched")
        total += count

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone. Total fetched: {total} real articles")
