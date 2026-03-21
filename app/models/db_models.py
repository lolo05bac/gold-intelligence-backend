"""
SQLAlchemy ORM models for the Gold Intelligence database.
"""
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean,
    Text, JSON, ForeignKey, Index, Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum


# ── Enums ──────────────────────────────────────────────────

class Tier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"


class Regime(str, enum.Enum):
    RISK_OFF = "risk_off"
    USD_DOMINANT = "usd_dominant"
    REAL_YIELD = "real_yield_driven"
    INFLATION_SCARE = "inflation_scare"
    FED_EVENT = "fed_event"
    GEOPOLITICAL = "geopolitical"
    RANGE_BOUND = "range_bound"
    TREND = "trend"


class Confidence(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Users ──────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255))
    tier = Column(SAEnum(Tier), default=Tier.FREE, nullable=False)
    stripe_customer_id = Column(String(255))
    stripe_subscription_id = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Market Data ────────────────────────────────────────────

class MarketData(Base):
    __tablename__ = "market_data"
    __table_args__ = (
        Index("ix_market_data_symbol_date", "symbol", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    source = Column(String(50))
    ingested_at = Column(DateTime, default=datetime.utcnow)


# ── Macro Releases ─────────────────────────────────────────

class MacroRelease(Base):
    __tablename__ = "macro_releases"
    __table_args__ = (
        Index("ix_macro_indicator_date", "indicator", "release_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator = Column(String(50), nullable=False)  # CPI, NFP, PCE, etc.
    release_date = Column(Date, nullable=False)
    actual = Column(Float)
    forecast = Column(Float)
    previous = Column(Float)
    surprise = Column(Float)  # actual - forecast
    surprise_std = Column(Float)  # standardized surprise
    source = Column(String(50))
    ingested_at = Column(DateTime, default=datetime.utcnow)


# ── Features (daily feature store) ─────────────────────────

class DailyFeatures(Base):
    __tablename__ = "daily_features"
    __table_args__ = (
        Index("ix_features_date", "date", unique=True),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    features = Column(JSON, nullable=False)  # All 42+ features as JSON
    feature_version = Column(String(20))
    computed_at = Column(DateTime, default=datetime.utcnow)


# ── Daily Signals (model outputs) ──────────────────────────

class DailySignal(Base):
    __tablename__ = "daily_signals"
    __table_args__ = (
        Index("ix_signal_date", "signal_date", unique=True),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_date = Column(Date, nullable=False, unique=True)

    # Core outputs
    bias_score = Column(Float, nullable=False)  # 1–10
    direction_probability = Column(Float)  # P(up day)
    expected_move_pct = Column(Float)  # signed expected move
    expected_range_pct = Column(Float)  # expected high-low range
    confidence = Column(Float)  # 0–1
    confidence_label = Column(SAEnum(Confidence))

    # Regime
    regime = Column(SAEnum(Regime))
    regime_probability = Column(Float)

    # Drivers (ranked list of explanations)
    bullish_drivers = Column(JSON)  # [{"name": "...", "impact": 0.x, "detail": "..."}, ...]
    bearish_drivers = Column(JSON)

    # Model metadata
    model_version = Column(String(20))
    feature_importance = Column(JSON)

    # Actuals (filled after market close)
    actual_return = Column(Float)
    actual_direction = Column(Boolean)  # True = up
    was_correct = Column(Boolean)

    computed_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


# ── News Sentiment ─────────────────────────────────────────

class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    __table_args__ = (
        Index("ix_news_date", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    hour = Column(Integer)  # 0–23 for intraday granularity

    # Sentiment scores
    geopolitical_tension = Column(Float, default=0)
    fed_hawkishness = Column(Float, default=0)
    fed_dovishness = Column(Float, default=0)
    inflation_scare = Column(Float, default=0)
    recession_fear = Column(Float, default=0)
    banking_stress = Column(Float, default=0)
    risk_off = Column(Float, default=0)
    safe_haven_demand = Column(Float, default=0)
    commodity_shock = Column(Float, default=0)

    # Raw counts
    total_articles = Column(Integer, default=0)
    gold_articles = Column(Integer, default=0)

    source = Column(String(50))
    computed_at = Column(DateTime, default=datetime.utcnow)


# ── Event Calendar ─────────────────────────────────────────

class EventCalendar(Base):
    __tablename__ = "event_calendar"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_date = Column(Date, nullable=False)
    event_time = Column(String(10))  # HH:MM ET
    event_name = Column(String(200), nullable=False)
    event_type = Column(String(50))  # fomc, cpi, nfp, speech, auction, geopolitical
    impact = Column(String(20))  # high, medium, low
    description = Column(Text)
    source = Column(String(50))


# ── Weekly Outlook ─────────────────────────────────────────

class WeeklyOutlook(Base):
    __tablename__ = "weekly_outlooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week_start = Column(Date, nullable=False)
    daily_forecasts = Column(JSON)  # [{day, bias, expected_move}, ...]
    narrative = Column(Text)
    risk_events = Column(JSON)
    computed_at = Column(DateTime, default=datetime.utcnow)


# ── Backtest Results ───────────────────────────────────────

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    model_version = Column(String(20))
    train_start = Column(Date)
    train_end = Column(Date)
    test_start = Column(Date)
    test_end = Column(Date)

    # Metrics
    accuracy = Column(Float)
    precision_bull = Column(Float)
    precision_bear = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    roc_auc = Column(Float)
    brier_score = Column(Float)
    high_conf_accuracy = Column(Float)

    # Regime-specific
    metrics_by_regime = Column(JSON)
    metrics_by_event = Column(JSON)

    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
