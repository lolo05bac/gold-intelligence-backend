"""
Application configuration loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_secret_key: str = "change-me"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/goldintel"
    redis_url: str = "redis://localhost:6379/0"

    # Data APIs
    twelve_data_api_key: str = ""
    polygon_api_key: str = ""
    fred_api_key: str = ""
    newsapi_key: str = ""
    alpha_vantage_key: str = ""

    # Auth
    jwt_secret_key: str = "change-me-jwt"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro: str = ""
    stripe_price_premium: str = ""

    # Model
    model_version: str = "2.4"
    prediction_horizon: str = "1d"
    confidence_threshold: float = 0.60
    high_confidence_threshold: float = 0.75

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
