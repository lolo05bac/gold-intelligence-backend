"""
Pydantic schemas for API request/response models.
"""
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ── Auth ───────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str]
    tier: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tier: str


# ── Signal / Dashboard ────────────────────────────────────

class DriverSchema(BaseModel):
    name: str
    impact: float = Field(ge=0, le=100)
    detail: str


class DailySignalResponse(BaseModel):
    signal_date: date
    bias_score: float = Field(ge=1, le=10)
    direction_probability: float = Field(ge=0, le=1)
    expected_move_pct: float
    expected_range_pct: Optional[float] = None
    confidence: float = Field(ge=0, le=1)
    confidence_label: str
    regime: str
    regime_probability: Optional[float] = None
    bullish_drivers: list[DriverSchema]
    bearish_drivers: list[DriverSchema]
    model_version: str
    actual_return: Optional[float] = None
    was_correct: Optional[bool] = None

    class Config:
        from_attributes = True


class DashboardResponse(BaseModel):
    """Full dashboard payload for frontend."""
    today: DailySignalResponse
    week_outlook: list[dict]
    recent_signals: list[DailySignalResponse]
    sentiment: dict
    events_today: list[dict]
    performance_30d: dict
    live_price: Optional[dict] = None


# ── Performance ────────────────────────────────────────────

class PerformanceMetrics(BaseModel):
    period: str
    accuracy: float
    high_conf_accuracy: float
    brier_score: float
    avg_return_per_signal: float
    total_signals: int
    correct_signals: int
    by_regime: dict
    by_event: dict


# ── Subscription ───────────────────────────────────────────

class SubscriptionCreate(BaseModel):
    price_id: str
    payment_method_id: Optional[str] = None


class SubscriptionResponse(BaseModel):
    subscription_id: str
    tier: str
    status: str
    current_period_end: Optional[datetime] = None
