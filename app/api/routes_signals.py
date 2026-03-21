"""
Signal API routes: daily bias, history, performance.
"""
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.db_models import DailySignal, Tier
from app.models.schemas import DailySignalResponse
from app.core.security import get_current_user

router = APIRouter()


@router.get("/today", response_model=DailySignalResponse)
async def get_today_signal(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get today's gold bias signal."""
    today = date.today()

    # Free tier gets yesterday's signal
    target_date = today
    if current_user["tier"] == "free":
        target_date = today - timedelta(days=1)

    result = await db.execute(
        select(DailySignal).where(DailySignal.signal_date == target_date)
    )
    signal = result.scalar_one_or_none()

    if not signal:
        raise HTTPException(status_code=404, detail=f"No signal for {target_date}")

    return signal


@router.get("/history", response_model=list[DailySignalResponse])
async def get_signal_history(
    days: int = Query(default=30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get historical signals."""
    # Free: 7 days, Pro: 90 days, Premium: 365 days
    max_days = {"free": 7, "pro": 90, "premium": 365}
    limit = min(days, max_days.get(current_user["tier"], 7))

    cutoff = date.today() - timedelta(days=limit)
    result = await db.execute(
        select(DailySignal)
        .where(DailySignal.signal_date >= cutoff)
        .order_by(desc(DailySignal.signal_date))
    )
    return result.scalars().all()


@router.get("/performance")
async def get_performance(
    period: str = Query(default="30d", regex="^(7d|30d|90d|1y)$"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get model performance metrics."""
    if current_user["tier"] == "free":
        raise HTTPException(status_code=403, detail="Performance metrics require Pro or Premium")

    days_map = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}
    cutoff = date.today() - timedelta(days=days_map[period])

    result = await db.execute(
        select(DailySignal)
        .where(DailySignal.signal_date >= cutoff)
        .where(DailySignal.actual_return.isnot(None))
        .order_by(DailySignal.signal_date)
    )
    signals = result.scalars().all()

    if not signals:
        return {"period": period, "message": "No evaluated signals yet"}

    total = len(signals)
    correct = sum(1 for s in signals if s.was_correct)
    high_conf = [s for s in signals if s.confidence >= 0.70]
    hc_correct = sum(1 for s in high_conf if s.was_correct)

    return {
        "period": period,
        "total_signals": total,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "high_conf_accuracy": round(hc_correct / len(high_conf), 4) if high_conf else None,
        "high_conf_count": len(high_conf),
        "avg_expected_move": round(sum(s.expected_move_pct or 0 for s in signals) / total, 4),
        "avg_actual_return": round(sum(s.actual_return or 0 for s in signals) / total, 5),
    }
