"""
Tests for the Gold Signal Engine.
Run: pytest tests/ -v
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFeatureBuilder:
    """Test feature engineering pipeline."""

    def test_rsi_bounds(self):
        """RSI should be between 0 and 100."""
        from pipelines.build_features import FeatureBuilder

        prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        rsi = FeatureBuilder._rsi(prices, 14)
        valid = rsi.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_atr_positive(self):
        """ATR should always be positive."""
        from pipelines.build_features import FeatureBuilder

        df = pd.DataFrame({
            "high": np.random.uniform(101, 105, 50),
            "low": np.random.uniform(95, 99, 50),
            "close": np.random.uniform(98, 103, 50),
        })
        atr = FeatureBuilder._atr(df, 14)
        valid = atr.dropna()
        assert (valid > 0).all()


class TestSignalEngine:
    """Test the signal scoring logic."""

    def test_bias_score_bounds(self):
        """Bias score should be between 1 and 10."""
        from pipelines.run_daily_model import GoldSignalEngine

        engine = GoldSignalEngine()

        # Test extreme bullish
        score = engine.compute_bias_score(0.95, 0.01, np.array([0.8, 0.1, 0.1]), 0.9)
        assert 1 <= score <= 10

        # Test extreme bearish
        score = engine.compute_bias_score(0.05, -0.01, np.array([0.8, 0.1, 0.1]), 0.9)
        assert 1 <= score <= 10

        # Test neutral
        score = engine.compute_bias_score(0.50, 0.0, np.array([0.33, 0.33, 0.34]), 0.5)
        assert 1 <= score <= 10

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        from pipelines.run_daily_model import GoldSignalEngine

        engine = GoldSignalEngine()

        # High agreement
        conf = engine.compute_confidence([0.7, 0.72, 0.68, 0.71], np.array([0.9, 0.05, 0.05]))
        assert 0 <= conf <= 1

        # Low agreement
        conf = engine.compute_confidence([0.3, 0.7, 0.5, 0.6], np.array([0.25, 0.25, 0.25, 0.25]))
        assert 0 <= conf <= 1

    def test_confidence_label(self):
        """Confidence labels should map correctly."""
        from pipelines.run_daily_model import GoldSignalEngine

        engine = GoldSignalEngine()
        assert engine.get_confidence_label(0.80) == "high"
        assert engine.get_confidence_label(0.55) == "medium"
        assert engine.get_confidence_label(0.30) == "low"

    def test_bullish_score_higher_than_bearish(self):
        """Higher P(up) should produce higher bias score."""
        from pipelines.run_daily_model import GoldSignalEngine

        engine = GoldSignalEngine()
        regime = np.array([0.5, 0.3, 0.2])

        bullish = engine.compute_bias_score(0.75, 0.005, regime, 0.7)
        bearish = engine.compute_bias_score(0.25, -0.005, regime, 0.7)
        assert bullish > bearish


class TestNewsSentiment:
    """Test news classification."""

    def test_classify_gold_article(self):
        """Gold-related articles should be detected."""
        from pipelines.ingest_news_data import NewsDataIngester

        ingester = NewsDataIngester()
        result = ingester.classify_article("Gold prices surge on falling real yields")
        assert result["is_gold_relevant"]

    def test_classify_geopolitical(self):
        """Geopolitical articles should be detected."""
        from pipelines.ingest_news_data import NewsDataIngester

        ingester = NewsDataIngester()
        result = ingester.classify_article("Military tensions escalate as sanctions imposed")
        assert result["is_geopolitical"]

    def test_classify_fed(self):
        """Fed articles should be detected."""
        from pipelines.ingest_news_data import NewsDataIngester

        ingester = NewsDataIngester()
        result = ingester.classify_article("Federal Reserve signals rate cut at next FOMC meeting")
        assert result["is_fed_relevant"]

    def test_sentiment_tone(self):
        """Positive articles should have positive tone."""
        from pipelines.ingest_news_data import NewsDataIngester

        ingester = NewsDataIngester()
        pos = ingester.classify_article("Markets rally strongly with major gains across sectors")
        neg = ingester.classify_article("Markets crash as panic selling leads to sharp decline")
        assert pos["tone"] > neg["tone"]


class TestMacroSurprise:
    """Test macro surprise computation."""

    def test_surprise_calculation(self):
        """Surprise should be actual minus forecast proxy."""
        from pipelines.ingest_macro_data import MacroDataIngester

        ingester = MacroDataIngester()
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 115],
        })
        result = ingester.compute_surprise(df, "level")
        # Last value (115) is much higher than recent trend — should have positive surprise
        assert result["surprise"].iloc[-1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
