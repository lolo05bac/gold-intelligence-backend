"""
GoldIntel.ai — FastAPI Application
Main entry point for the API server.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.db.database import init_db
from app.api.routes_auth import router as auth_router
from app.api.routes_signals import router as signals_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_billing import router as billing_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting GoldIntel.ai API...")
    if settings.app_debug:
        await init_db()
    yield
    logger.info("Shutting down GoldIntel.ai API...")


app = FastAPI(
    title="GoldIntel.ai API",
    description="Macro Intelligence Engine for Gold — Daily bias scores, expected moves, and driver analysis.",
    version=settings.model_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.app_debug else None,
    redoc_url="/redoc" if settings.app_debug else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://goldintel.ai",
        "https://www.goldintel.ai",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(signals_router, prefix="/api/signals", tags=["signals"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(billing_router, prefix="/api/billing", tags=["billing"])


@app.get("/")
async def root():
    return {
        "service": "GoldIntel.ai API",
        "version": settings.model_version,
        "status": "operational",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
