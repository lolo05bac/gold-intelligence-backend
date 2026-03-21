"""
Dashboard API routes: full dashboard payload, weekly outlook, events.
"""
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.db_models import DailySignal, WeeklyOutlook, EventCalendar, NewsSentiment
from app.models.schemas import DashboardResponse
from app.core.security import get_current_user

router = APIRouter()


@router.get("/full", response_model=DashboardResponse)
async def get_full_dashboard(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the complete dashboard payload."""
    today = date.today()
    is_free = current_user["tier"] == "free"

    # 1. Today's signal (or yesterday for free)
    target_date = today - timedelta(days=1) if is_free else today
    result = await db.execute(
        select(DailySignal).where(DailySignal.signal_date == target_date)
    )
    today_signal = result.scalar_one_or_none()
    if not today_signal:
        # Fall back to most recent signal
        result = await db.execute(
            select(DailySignal).order_by(desc(DailySignal.signal_date)).limit(1)
        )
        today_signal = result.scalar_one_or_none()

    if not today_signal:
        raise HTTPException(status_code=404, detail="No signals available")

    # 2. Weekly outlook
    week_outlook = []
    if not is_free:
        result = await db.execute(
            select(WeeklyOutlook)
            .order_by(desc(WeeklyOutlook.week_start))
            .limit(1)
        )
        outlook = result.scalar_one_or_none()
        if outlook and outlook.daily_forecasts:
            week_outlook = outlook.daily_forecasts

    # 3. Recent signals (last 10)
    history_days = 3 if is_free else 10
    cutoff = today - timedelta(days=30)
    result = await db.execute(
        select(DailySignal)
        .where(DailySignal.signal_date >= cutoff)
        .order_by(desc(DailySignal.signal_date))
        .limit(history_days)
    )
    recent_signals = result.scalars().all()

    # 4. Sentiment scores
    result = await db.execute(
        select(NewsSentiment)
        .where(NewsSentiment.date == today)
        .order_by(desc(NewsSentiment.computed_at))
        .limit(1)
    )
    sentiment_row = result.scalar_one_or_none()
    sentiment = {}
    if sentiment_row:
        sentiment = {
            "geopolitical_tension": sentiment_row.geopolitical_tension,
            "fed_hawkishness": sentiment_row.fed_hawkishness,
            "inflation_scare": sentiment_row.inflation_scare,
            "risk_off": sentiment_row.risk_off,
            "safe_haven_demand": sentiment_row.safe_haven_demand,
            "total_articles": sentiment_row.total_articles,
        }

    # 5. Today's events
    result = await db.execute(
        select(EventCalendar)
        .where(EventCalendar.event_date == today)
        .order_by(EventCalendar.event_time)
    )
    events = result.scalars().all()
    events_today = [
        {
            "time": e.event_time,
            "name": e.event_name,
            "type": e.event_type,
            "impact": e.impact,
        }
        for e in events
    ]

    # 6. 30-day performance summary
    perf_cutoff = today - timedelta(days=30)
    result = await db.execute(
        select(DailySignal)
        .where(DailySignal.signal_date >= perf_cutoff)
        .where(DailySignal.was_correct.isnot(None))
    )
    perf_signals = result.scalars().all()
    total_perf = len(perf_signals)
    correct_perf = sum(1 for s in perf_signals if s.was_correct)

    performance_30d = {
        "total": total_perf,
        "correct": correct_perf,
        "accuracy": round(correct_perf / total_perf, 4) if total_perf > 0 else 0,
    }

    return DashboardResponse(
        today=today_signal,
        week_outlook=week_outlook,
        recent_signals=recent_signals,
        sentiment=sentiment,
        events_today=events_today,
        performance_30d=performance_30d,
    )


@router.get("/events")
async def get_upcoming_events(
    days: int = 7,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get upcoming macro events."""
    today = date.today()
    end = today + timedelta(days=days)

    result = await db.execute(
        select(EventCalendar)
        .where(EventCalendar.event_date >= today)
        .where(EventCalendar.event_date <= end)
        .order_by(EventCalendar.event_date, EventCalendar.event_time)
    )
    events = result.scalars().all()
    return [
        {
            "date": str(e.event_date),
            "time": e.event_time,
            "name": e.event_name,
            "type": e.event_type,
            "impact": e.impact,
            "description": e.description,
        }
        for e in events
    ]


@router.get("/weekly-outlook")
async def get_weekly_outlook(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current weekly outlook."""
    if current_user["tier"] == "free":
        raise HTTPException(status_code=403, detail="Weekly outlook requires Pro or Premium")

    result = await db.execute(
        select(WeeklyOutlook)
        .order_by(desc(WeeklyOutlook.week_start))
        .limit(1)
    )
    outlook = result.scalar_one_or_none()
    if not outlook:
        raise HTTPException(status_code=404, detail="No weekly outlook available")

    return {
        "week_start": str(outlook.week_start),
        "daily_forecasts": outlook.daily_forecasts,
        "narrative": outlook.narrative,
        "risk_events": outlook.risk_events,
    }
