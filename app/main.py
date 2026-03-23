import os
import threading
import time
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="GoldIntel API", version="2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Live signal data (updated by background task)
current_signal = {
    "signal_date": datetime.now().strftime("%Y-%m-%d"),
    "weekly_bias": 7.8,
    "weekly_move": "+1.8%",
    "weekly_probability": 71,
    "weekly_confidence": 76,
    "daily_bias": 6.9,
    "daily_move": "+0.34%",
    "regime": "Geopolitical Risk-Bid / Falling Yields",
    "bullish_drivers": [
        {"name": "ME tensions sustaining safe-haven flows", "impact": 94},
        {"name": "Real yields declining", "impact": 89},
        {"name": "DXY weakening", "impact": 82},
    ],
    "bearish_drivers": [
        {"name": "Overbought on weekly RSI", "impact": 45},
        {"name": "PPI could surprise hawkish", "impact": 38},
    ],
    "last_updated": datetime.now().isoformat(),
}


def update_signal():
    """Fetch fresh data and update the signal."""
    global current_signal
    try:
        import yfinance as yf
        import numpy as np

        # Fetch latest gold data
        gold = yf.download("GC=F", period="60d", progress=False)
        dxy_data = yf.download("DX-Y.NYB", period="60d", progress=False)
        vix_data = yf.download("^VIX", period="60d", progress=False)
        spx_data = yf.download("^GSPC", period="60d", progress=False)
        oil_data = yf.download("CL=F", period="60d", progress=False)

        if len(gold) < 10:
            print("Not enough gold data")
            return

        # Calculate indicators
        gold_close = gold["Close"].values.flatten()
        gold_return_1d = (gold_close[-1] - gold_close[-2]) / gold_close[-2] * 100
        gold_return_5d = (gold_close[-1] - gold_close[-6]) / gold_close[-6] * 100
        gold_return_20d = (gold_close[-1] - gold_close[-21]) / gold_close[-21] * 100

        # RSI
        deltas = np.diff(gold_close[-15:])
        gains = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0
        losses = -np.mean([d for d in deltas if d < 0]) if any(d < 0 for d in deltas) else 0.001
        rsi = 100 - (100 / (1 + gains / losses))

        # DXY
        dxy_change = 0
        if len(dxy_data) >= 2:
            dxy_vals = dxy_data["Close"].values.flatten()
            dxy_change = (dxy_vals[-1] - dxy_vals[-2]) / dxy_vals[-2] * 100

        # VIX
        vix_level = 20
        vix_change = 0
        if len(vix_data) >= 2:
            vix_vals = vix_data["Close"].values.flatten()
            vix_level = float(vix_vals[-1])
            vix_change = float(vix_vals[-1] - vix_vals[-2])

        # SPX
        spx_return = 0
        if len(spx_data) >= 2:
            spx_vals = spx_data["Close"].values.flatten()
            spx_return = (spx_vals[-1] - spx_vals[-2]) / spx_vals[-2] * 100

        # Oil
        oil_return = 0
        if len(oil_data) >= 2:
            oil_vals = oil_data["Close"].values.flatten()
            oil_return = (oil_vals[-1] - oil_vals[-2]) / oil_vals[-2] * 100

        # Simple scoring logic
        score = 5.0  # neutral

        # Momentum
        if gold_return_5d > 1: score += 0.8
        elif gold_return_5d > 0: score += 0.3
        elif gold_return_5d < -1: score -= 0.8
        elif gold_return_5d < 0: score -= 0.3

        # DXY inverse
        if dxy_change < -0.3: score += 0.7
        elif dxy_change < 0: score += 0.3
        elif dxy_change > 0.3: score -= 0.7
        elif dxy_change > 0: score -= 0.3

        # VIX (fear = gold up)
        if vix_change > 2: score += 0.8
        elif vix_change > 0.5: score += 0.4
        elif vix_change < -2: score -= 0.5

        # Oil (inflation proxy)
        if oil_return > 2: score += 0.5
        elif oil_return < -2: score -= 0.3

        # SPX inverse (risk off = gold up)
        if spx_return < -1: score += 0.6
        elif spx_return > 1: score -= 0.3

        # RSI overbought/oversold
        if rsi > 70: score -= 0.4
        elif rsi < 30: score += 0.4

        # Clamp
        score = max(1.0, min(10.0, score))
        weekly_score = max(1.0, min(10.0, score + gold_return_5d * 0.3))

        # Build drivers
        bullish = []
        bearish = []

        if dxy_change < 0:
            bullish.append({"name": f"USD weakness (DXY {dxy_change:+.2f}%)", "impact": min(int(abs(dxy_change) * 100 + 50), 99)})
        else:
            bearish.append({"name": f"USD strength (DXY {dxy_change:+.2f}%)", "impact": min(int(abs(dxy_change) * 100 + 30), 99)})

        if vix_change > 0:
            bullish.append({"name": f"VIX rising ({vix_level:.1f}, +{vix_change:.1f})", "impact": min(int(vix_change * 15 + 50), 99)})
        else:
            bearish.append({"name": f"VIX declining ({vix_level:.1f}, {vix_change:.1f})", "impact": min(int(abs(vix_change) * 10 + 30), 99)})

        if gold_return_5d > 0:
            bullish.append({"name": f"5-day momentum positive ({gold_return_5d:+.2f}%)", "impact": min(int(gold_return_5d * 20 + 50), 99)})
        else:
            bearish.append({"name": f"5-day momentum negative ({gold_return_5d:+.2f}%)", "impact": min(int(abs(gold_return_5d) * 20 + 30), 99)})

        if spx_return < 0:
            bullish.append({"name": f"Equities falling — risk-off (SPX {spx_return:+.2f}%)", "impact": min(int(abs(spx_return) * 30 + 40), 99)})
        else:
            bearish.append({"name": f"Equities rising — risk-on (SPX {spx_return:+.2f}%)", "impact": min(int(spx_return * 20 + 25), 99)})

        if oil_return > 0:
            bullish.append({"name": f"Oil rising — inflation hedge ({oil_return:+.2f}%)", "impact": min(int(oil_return * 15 + 40), 99)})
        else:
            bearish.append({"name": f"Oil falling — deflation signal ({oil_return:+.2f}%)", "impact": min(int(abs(oil_return) * 10 + 25), 99)})

        if rsi > 65:
            bearish.append({"name": f"RSI overbought ({rsi:.0f})", "impact": min(int((rsi - 50) * 2), 99)})
        elif rsi < 35:
            bullish.append({"name": f"RSI oversold ({rsi:.0f})", "impact": min(int((50 - rsi) * 2), 99)})

        # Sort by impact
        bullish.sort(key=lambda x: x["impact"], reverse=True)
        bearish.sort(key=lambda x: x["impact"], reverse=True)

        # Determine regime
        if vix_change > 2 and spx_return < -0.5:
            regime = "Risk-Off / Flight to Safety"
        elif abs(dxy_change) > 0.3:
            regime = "USD-Dominant"
        elif gold_return_20d > 5:
            regime = "Trend / Momentum"
        elif oil_return > 1.5:
            regime = "Inflation Hedge"
        else:
            regime = "Mixed / Range-Bound"

        # Expected moves
        daily_move = f"{gold_return_1d:+.2f}%"
        weekly_move = f"{gold_return_5d:+.2f}%"

        prob = max(30, min(85, int(score * 8 + 10)))
        conf = max(25, min(90, int(abs(score - 5) * 15 + 40)))

        current_signal = {
            "signal_date": datetime.now().strftime("%Y-%m-%d"),
            "weekly_bias": round(weekly_score, 1),
            "weekly_move": weekly_move,
            "weekly_probability": prob,
            "weekly_confidence": conf,
            "daily_bias": round(score, 1),
            "daily_move": daily_move,
            "regime": regime,
            "bullish_drivers": bullish[:5],
            "bearish_drivers": bearish[:5],
            "last_updated": datetime.now().isoformat(),
            "gold_price": float(gold_close[-1]),
            "dxy": float(dxy_data["Close"].values.flatten()[-1]) if len(dxy_data) > 0 else 0,
            "vix": float(vix_level),
        }

        print(f"Signal updated: Bias {score:.1f}/10, Weekly {weekly_score:.1f}/10 at {datetime.now()}")

    except Exception as e:
        print(f"Update failed: {e}")


def background_updater():
    """Run updates every 10 minutes during market hours."""
    while True:
        try:
            update_signal()
        except Exception as e:
            print(f"Background update error: {e}")
        time.sleep(600)  # 10 minutes


@app.on_event("startup")
async def startup():
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()
    print("Background updater started — refreshing every 10 minutes")


@app.get("/")
async def root():
    return {"service": "GoldIntel API", "version": "2.4", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/signal/latest")
async def latest_signal():
    return current_signal

@app.get("/api/refresh")
async def manual_refresh():
    update_signal()
    return {"status": "refreshed", "signal": current_signal}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
