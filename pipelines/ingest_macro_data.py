"""
Macro Release Ingestion Pipeline
Fetches economic release data with surprise calculations.

Sources:
    - FRED: Historical macro series + release dates
    - BLS: CPI/PPI release schedule reference

Usage:
    python -m pipelines.ingest_macro_data [--start 2018-01-01]
"""
import os
import argparse
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import requests
from loguru import logger

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_RELEASE_BASE = "https://api.stlouisfed.org/fred/release/dates"

# Macro indicators we track, with their FRED series IDs
MACRO_INDICATORS = {
    "CPI": {
        "actual_series": "CPIAUCSL",      # CPI All Urban, SA
        "yoy_series": "CPIAUCNS",         # CPI All Urban, NSA (for YoY)
        "frequency": "monthly",
        "transform": "pct_change_yoy",
    },
    "Core_CPI": {
        "actual_series": "CPILFESL",      # CPI Less Food & Energy
        "frequency": "monthly",
        "transform": "pct_change_yoy",
    },
    "PPI": {
        "actual_series": "PPIACO",         # PPI All Commodities
        "frequency": "monthly",
        "transform": "pct_change_yoy",
    },
    "NFP": {
        "actual_series": "PAYEMS",         # Total Nonfarm Payrolls
        "frequency": "monthly",
        "transform": "diff",               # month-over-month change
    },
    "Unemployment": {
        "actual_series": "UNRATE",
        "frequency": "monthly",
        "transform": "level",
    },
    "PCE": {
        "actual_series": "PCEPI",          # PCE Price Index
        "frequency": "monthly",
        "transform": "pct_change_yoy",
    },
    "Core_PCE": {
        "actual_series": "PCEPILFE",
        "frequency": "monthly",
        "transform": "pct_change_yoy",
    },
    "ISM_Manufacturing": {
        "actual_series": "MANEMP",         # Proxy — ISM not directly on FRED
        "frequency": "monthly",
        "transform": "level",
    },
    "Retail_Sales": {
        "actual_series": "RSAFS",          # Retail Sales
        "frequency": "monthly",
        "transform": "pct_change_mom",
    },
    "GDP": {
        "actual_series": "GDP",
        "frequency": "quarterly",
        "transform": "pct_change_annualized",
    },
    "Initial_Claims": {
        "actual_series": "ICSA",           # Initial Jobless Claims
        "frequency": "weekly",
        "transform": "level",
    },
    "Consumer_Confidence": {
        "actual_series": "UMCSENT",        # U of Michigan Consumer Sentiment
        "frequency": "monthly",
        "transform": "level",
    },
}


class MacroDataIngester:
    """Fetches macro releases and computes surprise scores."""

    def __init__(self):
        self.fred_key = os.getenv("FRED_API_KEY", "")
        self.output_dir = os.path.join("data", "raw", "macro")
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_fred_series(self, series_id: str, start: str = "2016-01-01") -> pd.DataFrame:
        """Fetch a FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "observation_start": start,
            "sort_order": "asc",
        }
        try:
            resp = requests.get(FRED_BASE, params=params, timeout=30)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def compute_surprise(self, df: pd.DataFrame, transform: str) -> pd.DataFrame:
        """
        Compute 'surprise' as deviation from trailing forecast proxy.
        Since we don't have actual consensus, we use a rolling model as proxy:
        surprise = actual - rolling_mean(last 3 releases)
        """
        if df.empty:
            return df

        if transform == "pct_change_yoy":
            df["transformed"] = df["value"].pct_change(12) * 100  # 12-month YoY
        elif transform == "pct_change_mom":
            df["transformed"] = df["value"].pct_change() * 100
        elif transform == "diff":
            df["transformed"] = df["value"].diff()
        elif transform == "pct_change_annualized":
            df["transformed"] = df["value"].pct_change() * 400  # Annualized quarterly
        else:  # level
            df["transformed"] = df["value"]

        # Rolling "forecast" proxy: 3-period moving average of the transformed value
        df["forecast_proxy"] = df["transformed"].rolling(3, min_periods=1).mean().shift(1)
        df["surprise"] = df["transformed"] - df["forecast_proxy"]

        # Standardize surprise by rolling std
        rolling_std = df["surprise"].rolling(12, min_periods=3).std().shift(1)
        df["surprise_std"] = df["surprise"] / rolling_std.replace(0, np.nan)

        return df

    def run_full_ingestion(self, start_date: str = "2016-01-01"):
        """Ingest all macro indicators."""
        logger.info(f"=== Macro Data Ingestion from {start_date} ===")

        all_releases = []

        for indicator, config in MACRO_INDICATORS.items():
            logger.info(f"Processing {indicator}...")
            df = self.fetch_fred_series(config["actual_series"], start=start_date)

            if df.empty:
                logger.warning(f"  No data for {indicator}")
                continue

            df = self.compute_surprise(df, config["transform"])

            # Save individual series
            filepath = os.path.join(self.output_dir, f"{indicator.lower()}.parquet")
            df.to_parquet(filepath, index=False)
            logger.info(f"  → {len(df)} rows saved")

            # Append to master list
            for _, row in df.iterrows():
                if pd.notna(row.get("surprise")):
                    all_releases.append({
                        "indicator": indicator,
                        "release_date": row["date"],
                        "actual": row.get("transformed"),
                        "forecast": row.get("forecast_proxy"),
                        "surprise": row.get("surprise"),
                        "surprise_std": row.get("surprise_std"),
                    })

        # Save master release file
        releases_df = pd.DataFrame(all_releases)
        releases_df.to_parquet(
            os.path.join(self.output_dir, "all_macro_releases.parquet"),
            index=False,
        )
        logger.info(f"=== Macro ingestion complete: {len(releases_df)} total release observations ===")
        return releases_df


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    args = parser.parse_args()

    ingester = MacroDataIngester()
    ingester.run_full_ingestion(start_date=args.start)
