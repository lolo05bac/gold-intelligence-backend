"""
Billing routes: Stripe subscription management.
"""
import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.db_models import User, Tier
from app.models.schemas import SubscriptionCreate, SubscriptionResponse
from app.core.config import get_settings
from app.core.security import get_current_user

settings = get_settings()
stripe.api_key = settings.stripe_secret_key

router = APIRouter()


@router.post("/create-checkout")
async def create_checkout_session(
    payload: SubscriptionCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a Stripe Checkout session for subscription."""
    result = await db.execute(
        select(User).where(User.id == int(current_user["user_id"]))
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Create or retrieve Stripe customer
        if not user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.name,
                metadata={"user_id": str(user.id)},
            )
            user.stripe_customer_id = customer.id
            await db.flush()
        
        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=user.stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{"price": payload.price_id, "quantity": 1}],
            mode="subscription",
            success_url="https://goldintel.ai/dashboard?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://goldintel.ai/pricing",
            metadata={"user_id": str(user.id)},
        )

        return {"checkout_url": session.url, "session_id": session.id}

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid webhook")

    # Handle subscription events
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        subscription_id = session.get("subscription")

        if user_id and subscription_id:
            result = await db.execute(
                select(User).where(User.id == int(user_id))
            )
            user = result.scalar_one_or_none()
            if user:
                # Determine tier from price
                sub = stripe.Subscription.retrieve(subscription_id)
                price_id = sub["items"]["data"][0]["price"]["id"]

                if price_id == settings.stripe_price_premium:
                    user.tier = Tier.PREMIUM
                else:
                    user.tier = Tier.PRO

                user.stripe_subscription_id = subscription_id
                await db.flush()

    elif event["type"] in [
        "customer.subscription.deleted",
        "customer.subscription.paused",
    ]:
        sub = event["data"]["object"]
        sub_id = sub["id"]
        result = await db.execute(
            select(User).where(User.stripe_subscription_id == sub_id)
        )
        user = result.scalar_one_or_none()
        if user:
            user.tier = Tier.FREE
            user.stripe_subscription_id = None
            await db.flush()

    return {"status": "ok"}


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current subscription details."""
    result = await db.execute(
        select(User).where(User.id == int(current_user["user_id"]))
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.stripe_subscription_id:
        return SubscriptionResponse(
            subscription_id="",
            tier=user.tier.value,
            status="none",
        )

    try:
        sub = stripe.Subscription.retrieve(user.stripe_subscription_id)
        return SubscriptionResponse(
            subscription_id=sub.id,
            tier=user.tier.value,
            status=sub.status,
            current_period_end=sub.current_period_end,
        )
    except stripe.error.StripeError:
        return SubscriptionResponse(
            subscription_id=user.stripe_subscription_id,
            tier=user.tier.value,
            status="unknown",
        )


@router.post("/cancel")
async def cancel_subscription(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel the current subscription."""
    result = await db.execute(
        select(User).where(User.id == int(current_user["user_id"]))
    )
    user = result.scalar_one_or_none()
    if not user or not user.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")

    try:
        stripe.Subscription.modify(
            user.stripe_subscription_id,
            cancel_at_period_end=True,
        )
        return {"status": "cancellation_scheduled"}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
