# GoldIntel.ai — Macro Intelligence Engine for Gold

## Overview
Institutional-style gold intelligence platform that produces daily:
- **Gold Bias Score** (1–10): Probabilistic directional bias
- **Expected Move**: Predicted close-to-close and intraday range
- **Confidence Level**: 0–100% based on regime clarity and signal alignment
- **Key Drivers**: Ranked explanations of what's moving the model
- **Regime Detection**: Current market environment classification

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  Market Data │ Macro Data │ News/Sentiment │ Fed/Events  │
│  (Twelve/Poly) (FRED/BLS)  (NewsAPI/GDELT) (CME/Fed)   │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│                 FEATURE STORE                            │
│  Price/Technical │ Yields/USD │ Macro Surprise │ Sentiment│
│  ~42 features across 4 signal layers                     │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│                 SIGNAL ENGINE                             │
│  Regime Classifier → Direction Model → Move Model        │
│  → Event Shock Adjuster → Confidence Scorer              │
│  → Final Bias 1–10                                       │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│              API + DASHBOARD                              │
│  FastAPI Backend → Next.js Frontend                      │
│  PostgreSQL + Redis → Stripe Billing                     │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> && cd gold-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in API keys

# 3. Initialize database
alembic upgrade head

# 4. Run data ingestion
python -m pipelines.ingest_market_data
python -m pipelines.ingest_macro_data
python -m pipelines.ingest_news_data

# 5. Build features & run model
python -m pipelines.build_features
python -m pipelines.run_daily_model

# 6. Start API
uvicorn app.main:app --reload

# 7. Start frontend
cd frontend/nextjs-app && npm run dev
```

## Project Structure
```
gold-intelligence/
├── app/                  # FastAPI application
│   ├── api/              # Route handlers
│   ├── core/             # Config, security, deps
│   ├── models/           # SQLAlchemy + Pydantic models
│   ├── services/         # Business logic
│   ├── db/               # Database setup
│   └── main.py           # App entrypoint
├── data/                 # Data storage (gitignored)
├── research/             # Notebooks & experiments
├── pipelines/            # ETL & model pipelines
├── frontend/             # Next.js SaaS frontend
├── tests/                # Test suite
└── docker/               # Containerization
```

## Data Sources
| Layer | Source | Purpose |
|-------|--------|---------|
| Market | Twelve Data / Polygon | XAUUSD, DXY, yields, equities |
| Macro | FRED / BLS | CPI, NFP, PCE, GDP, ISM |
| News | NewsAPI + GDELT | Sentiment, geopolitical tension |
| Fed | Federal Reserve / CME | FOMC, speeches, rate expectations |

## Model Stack
1. **Regime Classifier** — XGBoost + HMM → identifies current environment
2. **Direction Model** — LightGBM ensemble → P(up day)
3. **Expected Move Model** — Quantile regression → magnitude
4. **Event Shock Adjuster** — Calibrated adjustments for CPI/FOMC/NFP
5. **Signal Combiner** — Weighted fusion → Bias 1–10

## Development Roadmap
- [x] Phase 1: Research build (features + baseline models)
- [ ] Phase 2: Production signal engine (automated daily runs)
- [ ] Phase 3: Professional website (SaaS dashboard)
- [ ] Phase 4: Monetization (Stripe subscriptions)
- [ ] Phase 5: Multi-asset expansion
