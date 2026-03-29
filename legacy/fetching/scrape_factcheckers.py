"""
Scrape fact-checker websites using their sitemaps.
Both sites explicitly allow all crawling in robots.txt (Disallow: nothing for content).
Stores scraped articles in PostgreSQL with label = 1 (fake/debunked).

Sources:
  - Factly.in         (India fact-checker, robots.txt: Allow all)
  - SM Hoaxslayer     (India fact-checker, robots.txt: Allow all)
  - Alt News          (India fact-checker, robots.txt: Disallow: nothing)
"""

import os
import re
import time
import urllib.request
import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Delay between requests — be polite to servers
REQUEST_DELAY = 0.5  # seconds

SOURCES = [
    {
        "name":    "Factly",
        "home":    "https://factly.in",
        "sitemap": "https://factly.in/sitemap_index.xml",
        "content_selectors": [".post-content", ".entry-content", "article"],
    },
    {
        "name":    "SM Hoaxslayer",
        "home":    "https://smhoaxslayer.com",
        "sitemap": "https://smhoaxslayer.com/sitemap_index.xml",
        "content_selectors": [".entry-content", ".post-content", "article"],
    },
    {
        "name":    "Alt News",
        "home":    "https://www.altnews.in",
        "sitemap": "https://www.altnews.in/sitemap_index.xml",
        "content_selectors": [".entry-content", ".post-content", "article"],
    },
]


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "fakenews"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )


def fetch_url(url, retries=2):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2)


def get_sitemap_urls(sitemap_url):
    """Recursively resolve sitemap index → individual article URLs."""
    try:
        content = fetch_url(sitemap_url).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"    Sitemap fetch failed: {e}")
        return []

    # If it's a sitemap index, recurse into sub-sitemaps
    sub_sitemaps = re.findall(r"<sitemap>\s*<loc>(.*?)</loc>", content, re.DOTALL)
    if sub_sitemaps:
        all_urls = []
        for sub in sub_sitemaps:
            sub = sub.strip()
            # Only process post sitemaps, skip category/tag sitemaps
            if any(skip in sub for skip in ["category", "tag", "author", "page"]):
                continue
            print(f"    Reading sub-sitemap: {sub}")
            all_urls.extend(get_sitemap_urls(sub))
            time.sleep(0.3)
        return all_urls

    # Leaf sitemap — extract article URLs
    urls = re.findall(r"<url>\s*<loc>(.*?)</loc>", content, re.DOTALL)
    return [u.strip() for u in urls]


def extract_article(html, selectors):
    """Extract title and content from article HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Remove noisy tags before extracting content
    for tag in soup.find_all(["script", "style", "nav", "header", "footer",
                               "aside", "figure", "figcaption", ".related",
                               ".share", ".social", ".comments", ".sidebar"]):
        tag.decompose()

    # Content
    content = ""
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            content = el.get_text(separator=" ", strip=True)
            if len(content) > 100:
                break

    return title, content


def get_existing_urls(cur):
    """Fetch all URLs already in the DB to skip re-scraping."""
    cur.execute("SELECT url FROM articles WHERE url IS NOT NULL")
    return set(row[0] for row in cur.fetchall())


def upsert_source(cur, name, url):
    cur.execute("""
        INSERT INTO sources (name, url, type, trust_level)
        VALUES (%s, %s, 'scrape', 'fake')
        ON CONFLICT (url) DO NOTHING
        RETURNING id
    """, (name, url))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("SELECT id FROM sources WHERE url = %s", (url,))
    return cur.fetchone()[0]


def scrape_source(source, existing_urls, conn):
    name       = source["name"]
    home       = source["home"]
    selectors  = source["content_selectors"]

    print(f"\n{'='*50}")
    print(f"Scraping: {name}")
    print(f"{'='*50}")

    cur = conn.cursor()
    source_id = upsert_source(cur, name, home)

    # Get all article URLs from sitemap
    print(f"  Resolving sitemaps...")
    urls = get_sitemap_urls(source["sitemap"])
    print(f"  Found {len(urls)} total URLs in sitemap")

    new_urls = [u for u in urls if u not in existing_urls]
    print(f"  {len(new_urls)} new (not yet in DB)")

    if not new_urls:
        print("  Nothing new to scrape.")
        cur.close()
        return 0

    batch = []
    saved = 0
    failed = 0

    for i, url in enumerate(new_urls, 1):
        try:
            html = fetch_url(url)
            title, content = extract_article(html, selectors)

            # Skip pages with too little content (likely not articles)
            if len(content) < 100:
                failed += 1
                continue

            batch.append((
                title[:500] if title else "",
                content,
                url,
                source_id,
                1,      # label = fake/debunked
                None,
                None,
            ))
            existing_urls.add(url)

            # Batch insert every 50 articles
            if len(batch) >= 50:
                execute_values(cur, """
                    INSERT INTO articles (title, content, url, source_id, label, subject, published_at)
                    VALUES %s ON CONFLICT (url) DO NOTHING
                """, batch)
                conn.commit()
                saved += len(batch)
                print(f"  [{i}/{len(new_urls)}] Saved batch — total so far: {saved}")
                batch = []

            time.sleep(REQUEST_DELAY)

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            break
        except Exception as e:
            failed += 1
            if failed % 50 == 0:
                print(f"  [{i}/{len(new_urls)}] {failed} failed so far (last: {e})")

    # Save remaining batch
    if batch:
        execute_values(cur, """
            INSERT INTO articles (title, content, url, source_id, label, subject, published_at)
            VALUES %s ON CONFLICT (url) DO NOTHING
        """, batch)
        conn.commit()
        saved += len(batch)

    # Log the fetch
    cur.execute("""
        INSERT INTO fetch_log (source_id, article_count, status)
        VALUES (%s, %s, 'success')
    """, (source_id, saved))
    conn.commit()
    cur.close()

    print(f"  Done — {saved} saved, {failed} skipped/failed")
    return saved


if __name__ == "__main__":
    conn = get_connection()
    cur  = conn.cursor()

    print("Loading existing URLs from DB...")
    existing_urls = get_existing_urls(cur)
    print(f"  {len(existing_urls)} articles already in DB")
    cur.close()

    total = 0
    for source in SOURCES:
        count = scrape_source(source, existing_urls, conn)
        total += count

    conn.close()
    print(f"\n{'='*50}")
    print(f"Scraping complete. Total new articles: {total}")
