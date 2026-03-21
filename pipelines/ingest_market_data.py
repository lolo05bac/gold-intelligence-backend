"""
Market Data Ingestion Pipeline
Fetches price data for gold, USD, yields, equities, and related instruments.

Sources:
    - Twelve Data (primary): XAUUSD, DXY, major FX, indices
    - FRED: Treasury yields, TIPS, breakevens
    - Polygon (backup): Futures, ETFs

Usage:
    python -m pipelines.ingest_market_data [--start 2018-01-01] [--end 2024-12-31]
"""
import os
import time
import argparse
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import requests
from loguru import logger

# ── Configuration ──────────────────────────────────────────

TWELVE_DATA_BASE = "https://api.twelvedata.com"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
POLYGON_BASE = "https://api.polygon.io/v2"

# Symbols to fetch from Twelve Data
MARKET_SYMBOLS = {
    # Gold & Precious Metals
    "XAU/USD": "gold_spot",
    "XAG/USD": "silver_spot",
    # Dollar & FX
    "USD/DXY": "dxy",       # Note: DXY may need special handling
    "EUR/USD": "eurusd",
    "USD/JPY": "usdjpy",
    "USD/CHF": "usdchf",
    # Equities
    "SPX": "spx",
    "NDX": "nasdaq",
    # Volatility
    "VIX": "vix",
    # Energy
    "CL": "crude_oil",
}

# FRED series for yields and rates
FRED_SERIES = {
    "DGS2": "us_2y_yield",
    "DGS5": "us_5y_yield",
    "DGS10": "us_10y_yield",
    "DGS30": "us_30y_yield",
    "DFII10": "us_10y_real_yield",  # 10Y TIPS
    "T10YIE": "us_10y_breakeven",
    "T5YIE": "us_5y_breakeven",
    "DFF": "fed_funds_rate",
    "TEDRATE": "ted_spread",
    "BAMLH0A0HYM2": "high_yield_spread",
}


class MarketDataIngester:
    """Fetches and stores historical + daily market data."""

    def __init__(self):
        self.twelve_key = os.getenv("TWELVE_DATA_API_KEY", "")
        self.fred_key = os.getenv("FRED_API_KEY", "")
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.output_dir = os.path.join("data", "raw", "market")
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Twelve Data ────────────────────────────────────────

    def fetch_twelve_data(
        self,
        symbol: str,
        interval: str = "1day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        outputsize: int = 5000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Twelve Data API."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.twelve_key,
            "outputsize": outputsize,
            "format": "JSON",
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        url = f"{TWELVE_DATA_BASE}/time_series"
        logger.info(f"Fetching {symbol} from Twelve Data...")

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "values" not in data:
                logger.warning(f"No values for {symbol}: {data.get('message', 'unknown error')}")
                return pd.DataFrame()

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("datetime").reset_index(drop=True)
            logger.info(f"  → {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # ── FRED ───────────────────────────────────────────────

    def fetch_fred_series(
        self,
        series_id: str,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch economic series from FRED."""
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
        if end_date:
            params["observation_end"] = end_date

        logger.info(f"Fetching FRED series {series_id}...")

        try:
            resp = requests.get(FRED_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            observations = data.get("observations", [])
            if not observations:
                logger.warning(f"No observations for {series_id}")
                return pd.DataFrame()

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "value"]].dropna()
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"  → {len(df)} rows for {series_id}")
            return df

        except Exception as e:
            logger.error(f"Error fetching FRED {series_id}: {e}")
            return pd.DataFrame()

    # ── Full Ingestion ─────────────────────────────────────

    def run_full_ingestion(
        self,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ):
        """Run complete market data ingestion."""
        if end_date is None:
            end_date = date.today().isoformat()

        logger.info(f"=== Market Data Ingestion: {start_date} to {end_date} ===")

        # 1. Twelve Data symbols
        market_frames = {}
        for symbol, name in MARKET_SYMBOLS.items():
            df = self.fetch_twelve_data(symbol, start_date=start_date, end_date=end_date)
            if not df.empty:
                df["symbol"] = name
                market_frames[name] = df
                filepath = os.path.join(self.output_dir, f"{name}.parquet")
                df.to_parquet(filepath, index=False)
            time.sleep(1)  # Rate limit courtesy

        # 2. FRED series
        fred_frames = {}
        for series_id, name in FRED_SERIES.items():
            df = self.fetch_fred_series(series_id, start_date=start_date, end_date=end_date)
            if not df.empty:
                df.columns = ["date", name]
                fred_frames[name] = df
                filepath = os.path.join(self.output_dir, f"fred_{name}.parquet")
                df.to_parquet(filepath, index=False)
            time.sleep(0.5)

        # 3. Merge all FRED into single yields file
        if fred_frames:
            merged = None
            for name, df in fred_frames.items():
                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on="date", how="outer")
            if merged is not None:
                merged = merged.sort_values("date").reset_index(drop=True)
                merged.to_parquet(os.path.join(self.output_dir, "fred_all.parquet"), index=False)
                logger.info(f"Merged FRED file: {len(merged)} rows, {len(merged.columns)} columns")

        logger.info("=== Market data ingestion complete ===")
        return market_frames, fred_frames

    def run_daily_update(self):
        """Fetch only today's data for live scoring."""
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=3)).isoformat()  # buffer for weekends
        return self.run_full_ingestion(start_date=yesterday, end_date=today)


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest market data")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--daily", action="store_true", help="Daily update mode")
    args = parser.parse_args()

    ingester = MarketDataIngester()
    if args.daily:
        ingester.run_daily_update()
    else:
        ingester.run_full_ingestion(start_date=args.start, end_date=args.end)
