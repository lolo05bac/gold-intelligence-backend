"""
Feature Engineering Pipeline
Builds the daily feature store from raw data across 4 signal layers.

Feature Layers:
    1. Price / Technical (15 features)
    2. Dollar / Yields (10 features)
    3. Macro Surprise (8 features)
    4. Risk / Sentiment (9 features)

Total: 42 features (Phase 1 MVP)

Usage:
    python -m pipelines.build_features [--start 2018-01-01]
"""
import os
import argparse
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

RAW_DIR = os.path.join("data", "raw")
FEATURES_DIR = os.path.join("data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)


class FeatureBuilder:
    """Builds the daily feature store from raw market/macro/news data."""

    def __init__(self):
        self.gold = None
        self.market = {}
        self.fred = None
        self.macro = None
        self.sentiment = None

    def load_data(self):
        """Load all raw data files."""
        logger.info("Loading raw data...")

        # Gold price
        gold_path = os.path.join(RAW_DIR, "market", "gold_spot.parquet")
        if os.path.exists(gold_path):
            self.gold = pd.read_parquet(gold_path)
            self.gold["date"] = pd.to_datetime(self.gold["datetime"]).dt.date
            logger.info(f"  Gold: {len(self.gold)} rows")

        # Other market symbols
        market_dir = os.path.join(RAW_DIR, "market")
        for f in os.listdir(market_dir):
            if f.endswith(".parquet") and f != "gold_spot.parquet" and not f.startswith("fred"):
                name = f.replace(".parquet", "")
                df = pd.read_parquet(os.path.join(market_dir, f))
                if "datetime" in df.columns:
                    df["date"] = pd.to_datetime(df["datetime"]).dt.date
                self.market[name] = df

        # FRED yields
        fred_path = os.path.join(RAW_DIR, "market", "fred_all.parquet")
        if os.path.exists(fred_path):
            self.fred = pd.read_parquet(fred_path)
            self.fred["date"] = pd.to_datetime(self.fred["date"]).dt.date
            logger.info(f"  FRED: {len(self.fred)} rows")

        # Macro releases
        macro_path = os.path.join(RAW_DIR, "macro", "all_macro_releases.parquet")
        if os.path.exists(macro_path):
            self.macro = pd.read_parquet(macro_path)
            logger.info(f"  Macro: {len(self.macro)} rows")

        # News sentiment
        news_path = os.path.join(RAW_DIR, "news", "daily_sentiment.parquet")
        if os.path.exists(news_path):
            self.sentiment = pd.read_parquet(news_path)
            logger.info(f"  Sentiment: {len(self.sentiment)} rows")

    # ═══════════════════════════════════════════════════════
    # LAYER 1: PRICE / TECHNICAL (15 features)
    # ═══════════════════════════════════════════════════════

    def build_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build technical/price features from gold OHLCV."""
        f = pd.DataFrame()
        f["date"] = df["date"]

        # Returns
        f["daily_return"] = df["close"].pct_change()
        f["overnight_return"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        f["intraday_return"] = (df["close"] - df["open"]) / df["open"]

        # Momentum
        f["momentum_5d"] = df["close"].pct_change(5)
        f["momentum_20d"] = df["close"].pct_change(20)
        f["momentum_60d"] = df["close"].pct_change(60)

        # Volatility
        f["atr_14"] = self._atr(df, 14) / df["close"]  # Normalized ATR
        f["realized_vol_20d"] = f["daily_return"].rolling(20).std() * np.sqrt(252)
        f["range_ratio"] = (df["high"] - df["low"]) / df["close"]

        # RSI
        f["rsi_14"] = self._rsi(df["close"], 14)

        # MACD slope
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        macd = ema12 - ema26
        f["macd_slope"] = macd - macd.shift(1)

        # Moving average relationship
        ma50 = df["close"].rolling(50).mean()
        ma200 = df["close"].rolling(200).mean()
        f["ma_50_slope"] = (ma50 - ma50.shift(5)) / ma50.shift(5)
        f["dist_from_ma50"] = (df["close"] - ma50) / ma50

        # Breakout signal
        f["breakout_20d"] = (df["close"] > df["high"].rolling(20).max().shift(1)).astype(int)

        return f

    # ═══════════════════════════════════════════════════════
    # LAYER 2: DOLLAR / YIELDS (10 features)
    # ═══════════════════════════════════════════════════════

    def build_dollar_yield_features(self) -> pd.DataFrame:
        """Build USD and rates features."""
        f = pd.DataFrame()

        # DXY features
        if "dxy" in self.market:
            dxy = self.market["dxy"]
            f["date"] = dxy["date"]
            f["dxy_return"] = dxy["close"].pct_change()
            f["dxy_5d_momentum"] = dxy["close"].pct_change(5)
        elif "eurusd" in self.market:
            # Use EURUSD as inverse proxy
            eur = self.market["eurusd"]
            f["date"] = eur["date"]
            f["dxy_return"] = -eur["close"].pct_change()  # inverse
            f["dxy_5d_momentum"] = -eur["close"].pct_change(5)

        # FX moves
        if "eurusd" in self.market:
            eur = self.market["eurusd"][["date","close"]].copy()
            eur["eurusd_return"] = eur["close"].pct_change()
            f = f.merge(eur[["date","eurusd_return"]], on="date", how="left")
            f["eurusd_return"] = f["eurusd_return"].ffill().fillna(0)
        if "usdjpy" in self.market:
            jpy = self.market["usdjpy"][["date","close"]].copy()
            jpy["usdjpy_return"] = jpy["close"].pct_change()
            f = f.merge(jpy[["date","usdjpy_return"]], on="date", how="left")
            f["usdjpy_return"] = f["usdjpy_return"].ffill().fillna(0)

        # Yield features from FRED
        if self.fred is not None:
            fred = self.fred.copy()
            fred["date"] = pd.to_datetime(fred["date"]).dt.date if not isinstance(fred["date"].iloc[0], date) else fred["date"]

            if "us_2y_yield" in fred.columns:
                f = f.merge(fred[["date", "us_2y_yield"]].rename(columns={"us_2y_yield": "yield_2y"}), on="date", how="left")
                f["yield_2y_change"] = f["yield_2y"].diff()

            if "us_10y_yield" in fred.columns:
                f = f.merge(fred[["date", "us_10y_yield"]].rename(columns={"us_10y_yield": "yield_10y"}), on="date", how="left")
                f["yield_10y_change"] = f["yield_10y"].diff()

            if "us_10y_real_yield" in fred.columns:
                f = f.merge(fred[["date", "us_10y_real_yield"]].rename(columns={"us_10y_real_yield": "real_yield_10y"}), on="date", how="left")
                f["real_yield_change"] = f["real_yield_10y"].diff()

            if "us_10y_breakeven" in fred.columns:
                f = f.merge(fred[["date", "us_10y_breakeven"]].rename(columns={"us_10y_breakeven": "breakeven_10y"}), on="date", how="left")
                f["breakeven_change"] = f["breakeven_10y"].diff()

            # 2s10s slope
            if "yield_2y" in f.columns and "yield_10y" in f.columns:
                f["curve_2s10s"] = f["yield_10y"] - f["yield_2y"]
                f["curve_2s10s_change"] = f["curve_2s10s"].diff()

        # Clean up intermediate columns
        drop_cols = ["yield_2y", "yield_10y", "real_yield_10y", "breakeven_10y", "curve_2s10s"]
        f = f.drop(columns=[c for c in drop_cols if c in f.columns], errors="ignore")

        return f

    # ═══════════════════════════════════════════════════════
    # LAYER 3: MACRO SURPRISE (8 features)
    # ═══════════════════════════════════════════════════════

    def build_macro_features(self, dates: pd.Series) -> pd.DataFrame:
        """Build macro surprise features aligned to trading dates."""
        f = pd.DataFrame({"date": dates})

        if self.macro is None or self.macro.empty:
            # Return empty columns
            for col in ["cpi_surprise", "nfp_surprise", "pce_surprise", "ism_surprise",
                        "retail_surprise", "claims_surprise", "unemployment_surprise", "gdp_surprise"]:
                f[col] = 0.0
            return f

        macro = self.macro.copy()
        macro["release_date"] = pd.to_datetime(macro["release_date"]).dt.date

        # For each indicator, get the most recent surprise as of each date
        key_indicators = {
            "CPI": "cpi_surprise",
            "NFP": "nfp_surprise",
            "Core_PCE": "pce_surprise",
            "ISM_Manufacturing": "ism_surprise",
            "Retail_Sales": "retail_surprise",
            "Initial_Claims": "claims_surprise",
            "Unemployment": "unemployment_surprise",
            "GDP": "gdp_surprise",
        }

        for indicator, col_name in key_indicators.items():
            ind_data = macro[macro["indicator"] == indicator][["release_date", "surprise_std"]].copy()
            ind_data = ind_data.rename(columns={"release_date": "date", "surprise_std": col_name})
            ind_data = ind_data.sort_values("date")

            # Forward-fill: carry the last surprise until the next release
            f = f.merge(ind_data, on="date", how="left")
            f[col_name] = f[col_name].ffill().fillna(0)

        return f

    # ═══════════════════════════════════════════════════════
    # LAYER 4: RISK / SENTIMENT (9 features)
    # ═══════════════════════════════════════════════════════

    def build_risk_sentiment_features(self, dates: pd.Series) -> pd.DataFrame:
        """Build risk and sentiment features."""
        f = pd.DataFrame({"date": dates})

        # VIX
        if "vix" in self.market:
            vix = self.market["vix"][["date", "close"]].rename(columns={"close": "vix_level"})
            f = f.merge(vix, on="date", how="left")
            f["vix_level"] = f["vix_level"].ffill()
            f["vix_change"] = f["vix_level"].pct_change()
        else:
            f["vix_level"] = 20
            f["vix_change"] = 0

        # SPX return
        if "spx" in self.market:
            spx = self.market["spx"][["date", "close"]].copy()
            spx["spx_return"] = spx["close"].pct_change()
            f = f.merge(spx[["date", "spx_return"]], on="date", how="left")
            f["spx_return"] = f["spx_return"].ffill().fillna(0)
        else:
            f["spx_return"] = 0

        # Oil return
        if "crude_oil" in self.market:
            oil = self.market["crude_oil"][["date", "close"]].copy()
            oil["oil_return"] = oil["close"].pct_change()
            f = f.merge(oil[["date", "oil_return"]], on="date", how="left")
            f["oil_return"] = f["oil_return"].ffill().fillna(0)
        else:
            f["oil_return"] = 0

        # Silver / gold ratio proxy
        if "silver_spot" in self.market:
            silver = self.market["silver_spot"][["date", "close"]].rename(columns={"close": "silver_close"})
            f = f.merge(silver, on="date", how="left")
            f["silver_close"] = f["silver_close"].ffill()
            f["silver_return"] = f["silver_close"].pct_change()
            f = f.drop(columns=["silver_close"])
        else:
            f["silver_return"] = 0

        # News sentiment scores
        if self.sentiment is not None and not self.sentiment.empty:
            sent = self.sentiment.copy()
            sent["date"] = pd.to_datetime(sent["date"]).dt.date if hasattr(sent["date"].iloc[0], "date") else sent["date"]
            sent_cols = ["geopolitical_tension", "fed_hawkishness", "inflation_scare",
                         "risk_off", "safe_haven_demand"]
            available = [c for c in sent_cols if c in sent.columns]
            if available:
                f = f.merge(sent[["date"] + available], on="date", how="left")
                for col in available:
                    f[col] = f[col].ffill().fillna(50)  # Default neutral
        else:
            for col in ["geopolitical_tension", "fed_hawkishness", "inflation_scare",
                        "risk_off", "safe_haven_demand"]:
                f[col] = 50

        # Drop vix_level (keep vix_change)
        f = f.drop(columns=["vix_level"], errors="ignore")

        return f

    # ═══════════════════════════════════════════════════════
    # TARGET VARIABLE
    # ═══════════════════════════════════════════════════════

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build prediction targets: next-day direction and return."""
        t = pd.DataFrame()
        t["date"] = df["date"]

        # Next day return (close-to-close)
        t["target_return"] = df["close"].pct_change().shift(-1)

        # Next day direction (1 = up, 0 = down)
        t["target_direction"] = (t["target_return"] > 0).astype(int)

        # Next day absolute return
        t["target_abs_return"] = t["target_return"].abs()

        # Next day range (high-low / close)
        t["target_range"] = ((df["high"].shift(-1) - df["low"].shift(-1)) / df["close"]).values

        return t

    # ═══════════════════════════════════════════════════════
    # MASTER BUILD
    # ═══════════════════════════════════════════════════════

    def build_all(self, start_date: str = "2018-01-01") -> pd.DataFrame:
        """Build the complete daily feature store."""
        self.load_data()

        if self.gold is None or self.gold.empty:
            logger.error("No gold price data found. Run ingest_market_data first.")
            return pd.DataFrame()

        logger.info("=== Building Feature Store ===")

        # Layer 1: Price / Technical
        price_feat = self.build_price_features(self.gold)
        logger.info(f"  Layer 1 (Price/Tech): {len(price_feat.columns) - 1} features")

        # Layer 2: Dollar / Yields
        dy_feat = self.build_dollar_yield_features()
        logger.info(f"  Layer 2 (Dollar/Yields): {len(dy_feat.columns) - 1} features")

        # Layer 3: Macro Surprises
        macro_feat = self.build_macro_features(price_feat["date"])
        logger.info(f"  Layer 3 (Macro): {len(macro_feat.columns) - 1} features")

        # Layer 4: Risk / Sentiment
        risk_feat = self.build_risk_sentiment_features(price_feat["date"])
        logger.info(f"  Layer 4 (Risk/Sentiment): {len(risk_feat.columns) - 1} features")

        # Targets
        targets = self.build_targets(self.gold)

        # Merge all layers
        features = price_feat
        for df in [dy_feat, macro_feat, risk_feat, targets]:
            features = features.merge(df, on="date", how="left")

        # Filter date range
        start = pd.to_datetime(start_date).date()
        features = features[features["date"] >= start].copy()

        # Forward fill then drop remaining NaN rows
        feature_cols = [c for c in features.columns if c not in ["date", "target_return", "target_direction", "target_abs_return", "target_range"]]
        features[feature_cols] = features[feature_cols].ffill()
        features = features.dropna(subset=feature_cols, how="any")

        # Save
        features.to_parquet(os.path.join(FEATURES_DIR, "daily_features.parquet"), index=False)

        total_features = len(feature_cols)
        logger.info(f"=== Feature store built: {len(features)} rows × {total_features} features ===")
        logger.info(f"    Feature columns: {feature_cols}")

        return features

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    args = parser.parse_args()

    builder = FeatureBuilder()
    df = builder.build_all(start_date=args.start)
    print(f"\nFeature store shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
