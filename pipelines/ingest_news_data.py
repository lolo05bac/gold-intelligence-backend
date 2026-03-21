"""
News & Sentiment Ingestion Pipeline
Fetches news articles and computes structured sentiment scores.

Sources:
    - NewsAPI: Article flow and headlines
    - GDELT: Event database and tone analysis

Usage:
    python -m pipelines.ingest_news_data [--days 30]
"""
import os
import re
import argparse
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import requests
from loguru import logger

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TV_API = "https://api.gdeltproject.org/api/v2/tv/tv"

# Keyword sets for topic classification
TOPIC_KEYWORDS = {
    "gold": ["gold", "xauusd", "gold price", "bullion", "precious metal", "gold futures", "comex gold"],
    "inflation": ["inflation", "cpi", "consumer prices", "price index", "cost of living", "pce"],
    "fed": ["federal reserve", "fed", "fomc", "powell", "rate hike", "rate cut", "monetary policy", "interest rate"],
    "geopolitical": ["war", "conflict", "sanctions", "military", "missile", "attack", "invasion", "nuclear", "tension"],
    "recession": ["recession", "slowdown", "downturn", "contraction", "economic decline", "gdp decline"],
    "banking": ["bank failure", "banking crisis", "bank run", "credit crunch", "financial stress", "liquidity"],
    "risk_off": ["safe haven", "flight to safety", "risk aversion", "market crash", "sell-off", "panic"],
    "oil": ["oil price", "crude oil", "opec", "petroleum", "energy prices", "oil supply"],
    "usd": ["dollar", "usd", "dxy", "dollar index", "greenback", "dollar strength", "dollar weakness"],
}

# Simple sentiment words for tone scoring
POSITIVE_WORDS = {"surge", "rally", "gain", "soar", "jump", "rise", "bullish", "strong", "boom", "recovery"}
NEGATIVE_WORDS = {"crash", "plunge", "fall", "drop", "decline", "bearish", "weak", "slump", "collapse", "fear"}
HAWKISH_WORDS = {"hike", "tighten", "hawkish", "restrictive", "inflation fight", "higher for longer"}
DOVISH_WORDS = {"cut", "ease", "dovish", "accommodative", "pivot", "pause", "lower rates"}


class NewsDataIngester:
    """Fetches news and builds structured sentiment scores."""

    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.output_dir = os.path.join("data", "raw", "news")
        os.makedirs(self.output_dir, exist_ok=True)

    # ── NewsAPI ────────────────────────────────────────────

    def fetch_newsapi(
        self,
        query: str,
        from_date: str,
        to_date: str,
        page_size: int = 100,
        language: str = "en",
    ) -> list[dict]:
        """Fetch articles from NewsAPI."""
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": self.newsapi_key,
        }
        try:
            resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            logger.info(f"  NewsAPI '{query}': {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"NewsAPI error for '{query}': {e}")
            return []

    # ── GDELT ──────────────────────────────────────────────

    def fetch_gdelt_articles(
        self,
        query: str,
        mode: str = "artlist",
        timespan: str = "1440",  # minutes (24h)
        max_records: int = 250,
    ) -> list[dict]:
        """Fetch articles from GDELT DOC API."""
        params = {
            "query": query,
            "mode": mode,
            "format": "json",
            "timespan": f"{timespan}min",
            "maxrecords": max_records,
            "sort": "datedesc",
        }
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            logger.info(f"  GDELT '{query}': {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"GDELT error for '{query}': {e}")
            return []

    # ── Sentiment Scoring ──────────────────────────────────

    def classify_article(self, title: str, description: str = "") -> dict:
        """Classify a single article by topic and sentiment."""
        text = f"{title} {description}".lower()

        # Topic matching
        topics = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                topics[topic] = min(score / len(keywords), 1.0)

        # Simple sentiment
        words = set(re.findall(r'\w+', text))
        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)
        hawk_count = len(words & HAWKISH_WORDS)
        dove_count = len(words & DOVISH_WORDS)

        total_sentiment = pos_count + neg_count
        tone = 0
        if total_sentiment > 0:
            tone = (pos_count - neg_count) / total_sentiment  # -1 to +1

        fed_tone = 0
        total_fed = hawk_count + dove_count
        if total_fed > 0:
            fed_tone = (hawk_count - dove_count) / total_fed  # positive = hawkish

        return {
            "topics": topics,
            "tone": tone,
            "fed_tone": fed_tone,
            "is_gold_relevant": "gold" in topics,
            "is_geopolitical": "geopolitical" in topics,
            "is_fed_relevant": "fed" in topics,
        }

    def build_daily_sentiment(self, articles: list[dict], target_date: date) -> dict:
        """Aggregate article-level scores into daily sentiment scores."""
        if not articles:
            return self._empty_sentiment(target_date)

        classifications = []
        for art in articles:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            cls = self.classify_article(title, desc)
            classifications.append(cls)

        n = len(classifications)

        # Aggregate topic intensities
        def avg_topic(topic_key):
            vals = [c["topics"].get(topic_key, 0) for c in classifications]
            return np.mean(vals) * 100 if vals else 0

        # Build sentiment scores (0–100 scale)
        sentiment = {
            "date": target_date,
            "total_articles": n,
            "gold_articles": sum(1 for c in classifications if c["is_gold_relevant"]),
            "geopolitical_tension": avg_topic("geopolitical"),
            "fed_hawkishness": np.mean([c["fed_tone"] for c in classifications if c["is_fed_relevant"]] or [0]) * 50 + 50,
            "fed_dovishness": 100 - (np.mean([c["fed_tone"] for c in classifications if c["is_fed_relevant"]] or [0]) * 50 + 50),
            "inflation_scare": avg_topic("inflation"),
            "recession_fear": avg_topic("recession"),
            "banking_stress": avg_topic("banking"),
            "risk_off": avg_topic("risk_off"),
            "safe_haven_demand": (avg_topic("risk_off") + avg_topic("gold")) / 2,
            "commodity_shock": avg_topic("oil"),
            "overall_tone": np.mean([c["tone"] for c in classifications]),
        }
        return sentiment

    def _empty_sentiment(self, target_date: date) -> dict:
        return {
            "date": target_date,
            "total_articles": 0, "gold_articles": 0,
            "geopolitical_tension": 50, "fed_hawkishness": 50, "fed_dovishness": 50,
            "inflation_scare": 50, "recession_fear": 50, "banking_stress": 50,
            "risk_off": 50, "safe_haven_demand": 50, "commodity_shock": 50,
            "overall_tone": 0,
        }

    # ── Full Ingestion ─────────────────────────────────────

    def run_ingestion(self, days_back: int = 30):
        """Run news ingestion for the last N days."""
        logger.info(f"=== News Ingestion: last {days_back} days ===")

        all_sentiments = []
        today = date.today()

        for day_offset in range(days_back):
            target = today - timedelta(days=day_offset)
            from_date = target.isoformat()
            to_date = (target + timedelta(days=1)).isoformat()

            logger.info(f"Processing {from_date}...")

            # Fetch from multiple queries
            articles = []
            for query in ["gold price", "federal reserve", "inflation", "geopolitical"]:
                batch = self.fetch_newsapi(query, from_date, to_date, page_size=50)
                articles.extend(batch)

            # Deduplicate by title
            seen_titles = set()
            unique = []
            for art in articles:
                title = (art.get("title") or "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique.append(art)

            sentiment = self.build_daily_sentiment(unique, target)
            all_sentiments.append(sentiment)

        # Save
        df = pd.DataFrame(all_sentiments)
        df.to_parquet(os.path.join(self.output_dir, "daily_sentiment.parquet"), index=False)
        logger.info(f"=== News ingestion complete: {len(df)} daily records ===")
        return df

    def run_daily_update(self):
        """Fetch only today's news for live scoring."""
        return self.run_ingestion(days_back=1)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    ingester = NewsDataIngester()
    ingester.run_ingestion(days_back=args.days)
