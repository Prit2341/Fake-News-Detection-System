"""
Fetch live news articles from NewsAPI and save to local cache.
Reads API key from .env file.
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2"
CACHE_DIR = Path("d:/Fake_News_Detection/cache")
CACHE_DIR.mkdir(exist_ok=True)

CATEGORIES = ["politics", "technology", "health", "world", "science"]


def fetch_top_headlines(category: str) -> list[dict]:
    """Fetch top headlines for a given category."""
    url = f"{BASE_URL}/top-headlines"
    params = {
        "category": category,
        "language": "en",
        "pageSize": 100,
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        print(f"  API error for category '{category}': {data.get('message')}")
        return []

    articles = data.get("articles", [])
    print(f"  [{category}] Fetched {len(articles)} articles")
    return articles


def fetch_by_keyword(keyword: str) -> list[dict]:
    """Search articles by keyword."""
    url = f"{BASE_URL}/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        print(f"  API error for keyword '{keyword}': {data.get('message')}")
        return []

    articles = data.get("articles", [])
    print(f"  [keyword: {keyword}] Fetched {len(articles)} articles")
    return articles


def save_to_cache(articles: list[dict], name: str):
    """Save fetched articles to a JSON cache file."""
    cache_file = CACHE_DIR / f"{name}.json"
    payload = {
        "fetched_at": datetime.utcnow().isoformat(),
        "count": len(articles),
        "articles": articles,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(articles)} articles → cache/{name}.json")


def main():
    if not API_KEY or API_KEY == "your_api_key_here":
        print("ERROR: NEWS_API_KEY not set in .env file.")
        return

    print(f"NewsAPI Key loaded: {API_KEY[:6]}{'*' * (len(API_KEY) - 6)}")
    print(f"Cache directory: {CACHE_DIR}\n")

    # Fetch top headlines per category
    print("Fetching top headlines by category...")
    all_articles = []
    for category in CATEGORIES:
        articles = fetch_top_headlines(category)
        save_to_cache(articles, f"headlines_{category}")
        all_articles.extend(articles)

    # Save combined
    save_to_cache(all_articles, "headlines_all")
    print(f"\nTotal articles fetched: {len(all_articles)}")
    print("\nDone. Check cache/ folder for results.")


if __name__ == "__main__":
    main()
