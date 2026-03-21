from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GoldIntel API", version="2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"service": "GoldIntel API", "version": "2.4", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/signal/latest")
async def latest_signal():
    return {
        "signal_date": "2026-03-21",
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
            {"name": "DXY weakening toward 99.0", "impact": 82},
        ],
        "bearish_drivers": [
            {"name": "Overbought on weekly RSI", "impact": 45},
            {"name": "PPI could surprise hawkish", "impact": 38},
        ],
    }
