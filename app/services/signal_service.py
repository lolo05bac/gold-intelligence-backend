"""
Signal Service — Orchestrates the daily pipeline.
Called by cron scheduler or API trigger to produce the daily signal.
"""
import os
import json
from datetime import date, datetime
from typing import Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.db_models import DailySignal, DailyFeatures, NewsSentiment, Confidence, Regime


class SignalService:
    """Orchestrates data refresh, feature build, model scoring, and DB storage."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def run_daily_pipeline(self, target_date: Optional[date] = None) -> dict:
        """
        Full daily pipeline:
        1. Refresh data (market, macro, news)
        2. Build features
        3. Run model
        4. Store signal in database
        5. Return signal dict
        """
        if target_date is None:
            target_date = date.today()

        logger.info(f"=== Running daily pipeline for {target_date} ===")

        # Step 1: Data refresh
        await self._refresh_data()

        # Step 2: Build features
        await self._build_features()

        # Step 3: Run model
        signal = await self._run_model(target_date)

        # Step 4: Store in database
        await self._store_signal(signal)

        logger.info(f"=== Pipeline complete: Bias {signal['bias_score']}/10 ===")
        return signal

    async def _refresh_data(self):
        """Trigger data ingestion pipelines."""
        try:
            from pipelines.ingest_market_data import MarketDataIngester
            from pipelines.ingest_news_data import NewsDataIngester

            market = MarketDataIngester()
            market.run_daily_update()

            news = NewsDataIngester()
            news.run_daily_update()

            logger.info("Data refresh complete")
        except Exception as e:
            logger.error(f"Data refresh error: {e}")
            # Continue with stale data rather than failing

    async def _build_features(self):
        """Run feature engineering."""
        try:
            from pipelines.build_features import FeatureBuilder
            builder = FeatureBuilder()
            builder.build_all()
            logger.info("Feature build complete")
        except Exception as e:
            logger.error(f"Feature build error: {e}")
            raise

    async def _run_model(self, target_date: date) -> dict:
        """Score today with the signal engine."""
        from pipelines.run_daily_model import GoldSignalEngine
        engine = GoldSignalEngine()
        signal = engine.score_today(target_date=target_date)
        return signal

    async def _store_signal(self, signal: dict):
        """Persist the daily signal to the database."""
        signal_date = signal["signal_date"]

        # Check if signal already exists
        result = await self.db.execute(
            select(DailySignal).where(DailySignal.signal_date == signal_date)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update
            existing.bias_score = signal["bias_score"]
            existing.direction_probability = signal["direction_probability"]
            existing.expected_move_pct = signal["expected_move_pct"]
            existing.confidence = signal["confidence"]
            existing.confidence_label = Confidence(signal["confidence_label"])
            existing.regime = Regime(signal["regime"])
            existing.regime_probability = signal.get("regime_probability")
            existing.bullish_drivers = signal.get("bullish_drivers", [])
            existing.bearish_drivers = signal.get("bearish_drivers", [])
            existing.model_version = signal.get("model_version", "2.4")
            existing.feature_importance = signal.get("individual_probs")
            logger.info(f"Updated existing signal for {signal_date}")
        else:
            # Insert
            db_signal = DailySignal(
                signal_date=signal_date,
                bias_score=signal["bias_score"],
                direction_probability=signal["direction_probability"],
                expected_move_pct=signal["expected_move_pct"],
                confidence=signal["confidence"],
                confidence_label=Confidence(signal["confidence_label"]),
                regime=Regime(signal["regime"]),
                regime_probability=signal.get("regime_probability"),
                bullish_drivers=signal.get("bullish_drivers", []),
                bearish_drivers=signal.get("bearish_drivers", []),
                model_version=signal.get("model_version", "2.4"),
                feature_importance=signal.get("individual_probs"),
            )
            self.db.add(db_signal)
            logger.info(f"Stored new signal for {signal_date}")

        await self.db.flush()

    async def backfill_actuals(self, signal_date: date, actual_return: float):
        """
        Fill in actual results after market close.
        Called by a separate cron job after NY close.
        """
        result = await self.db.execute(
            select(DailySignal).where(DailySignal.signal_date == signal_date)
        )
        signal = result.scalar_one_or_none()

        if signal:
            signal.actual_return = actual_return
            signal.actual_direction = actual_return > 0

            # Was the direction call correct?
            predicted_up = signal.direction_probability > 0.5
            signal.was_correct = predicted_up == (actual_return > 0)

            await self.db.flush()
            logger.info(
                f"Backfilled actuals for {signal_date}: "
                f"return={actual_return:.4f}, correct={signal.was_correct}"
            )
