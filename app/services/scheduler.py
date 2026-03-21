"""
Scheduler — Automated daily pipeline execution.
Uses APScheduler for cron-like scheduling.

Schedule:
    - 07:00 ET: Run daily signal pipeline (pre-market)
    - 17:00 ET: Backfill actual results (post-market)
    - 06:00 ET: Refresh data feeds

For Railway: run as a separate worker service.
"""
import asyncio
from datetime import date, timedelta, timezone, datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from app.db.database import async_session
from app.services.signal_service import SignalService


# Eastern Time offset (UTC-5, or UTC-4 during DST)
ET_OFFSET = -5


async def run_daily_signal():
    """Generate today's gold bias signal. Runs pre-market."""
    logger.info("⏰ CRON: Running daily signal pipeline...")
    try:
        async with async_session() as db:
            service = SignalService(db)
            signal = await service.run_daily_pipeline()
            await db.commit()
            logger.info(f"⏰ CRON: Signal complete — Bias {signal['bias_score']}/10")
    except Exception as e:
        logger.error(f"⏰ CRON: Daily signal failed: {e}")


async def backfill_actuals():
    """Fill in actual market results. Runs after market close."""
    logger.info("⏰ CRON: Backfilling actual results...")
    try:
        # Get yesterday's actual gold return
        from pipelines.ingest_market_data import MarketDataIngester
        import pandas as pd

        ingester = MarketDataIngester()
        df = ingester.fetch_twelve_data("XAU/USD", interval="1day", outputsize=5)

        if df is not None and len(df) >= 2:
            # Most recent completed day
            today_close = float(df.iloc[-1]["close"])
            prev_close = float(df.iloc[-2]["close"])
            actual_return = (today_close - prev_close) / prev_close

            signal_date = date.today()

            async with async_session() as db:
                service = SignalService(db)
                await service.backfill_actuals(signal_date, actual_return)
                await db.commit()
                logger.info(f"⏰ CRON: Backfill complete — return={actual_return:.4f}")
        else:
            logger.warning("⏰ CRON: Insufficient price data for backfill")

    except Exception as e:
        logger.error(f"⏰ CRON: Backfill failed: {e}")


async def refresh_data_feeds():
    """Pre-dawn data refresh. Ensures fresh data for morning run."""
    logger.info("⏰ CRON: Refreshing data feeds...")
    try:
        from pipelines.ingest_market_data import MarketDataIngester
        from pipelines.ingest_macro_data import MacroDataIngester
        from pipelines.ingest_news_data import NewsDataIngester

        MarketDataIngester().run_daily_update()
        MacroDataIngester().run_full_ingestion()
        NewsDataIngester().run_daily_update()

        logger.info("⏰ CRON: Data refresh complete")
    except Exception as e:
        logger.error(f"⏰ CRON: Data refresh failed: {e}")


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the scheduler."""
    scheduler = AsyncIOScheduler(timezone="US/Eastern")

    # 06:00 ET — Refresh all data feeds
    scheduler.add_job(
        refresh_data_feeds,
        CronTrigger(hour=6, minute=0, timezone="US/Eastern"),
        id="refresh_data",
        name="Refresh data feeds",
        replace_existing=True,
    )

    # 07:00 ET — Generate daily signal (before US pre-market)
    scheduler.add_job(
        run_daily_signal,
        CronTrigger(hour=7, minute=0, day_of_week="mon-fri", timezone="US/Eastern"),
        id="daily_signal",
        name="Daily gold signal",
        replace_existing=True,
    )

    # 17:00 ET — Backfill actuals (after NYSE close)
    scheduler.add_job(
        backfill_actuals,
        CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone="US/Eastern"),
        id="backfill_actuals",
        name="Backfill actual results",
        replace_existing=True,
    )

    return scheduler


if __name__ == "__main__":
    """Run the scheduler standalone (for Railway worker service)."""
    logger.info("Starting GoldIntel scheduler worker...")

    scheduler = create_scheduler()
    scheduler.start()

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped")
