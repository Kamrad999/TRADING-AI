"""Configuration management for AMATIS.

Uses Pydantic Settings for environment-based configuration with:
    - Type safety
    - Validation
    - Environment variable loading
    - Secrets management preparation
    - Hierarchical configuration
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    TEXT = "text"
    COLORED = "colored"


class RiskConfig(BaseSettings):
    """Risk management configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AMATIX_RISK_")
    
    max_portfolio_exposure: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Maximum portfolio exposure as fraction of capital",
    )
    max_daily_drawdown: float = Field(
        default=0.025,
        ge=0.0,
        le=1.0,
        description="Maximum daily drawdown before kill switch",
    )
    max_risk_per_trade: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Maximum risk per trade as fraction of capital",
    )
    max_open_positions: int = Field(
        default=5,
        ge=1,
        description="Maximum number of concurrent positions",
    )
    max_ticker_exposure: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum exposure to single ticker",
    )
    kill_switch_enabled: bool = Field(
        default=True,
        description="Enable automatic kill switch on risk limits",
    )
    paper_mode: bool = Field(
        default=True,
        description="Run in paper trading mode (no real orders)",
    )
    
    @model_validator(mode="after")
    def validate_risk_limits(self) -> RiskConfig:
        """Ensure risk limits are internally consistent."""
        if self.max_risk_per_trade > self.max_portfolio_exposure:
            raise ValueError("max_risk_per_trade cannot exceed max_portfolio_exposure")
        return self


class ExecutionConfig(BaseSettings):
    """Order execution configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AMATIX_EXEC_")
    
    default_broker: str = Field(default="paper")
    valid_brokers: Set[str] = Field(default={"paper", "alpaca", "ibkr", "binance"})
    max_slippage_bps: float = Field(default=10.0, ge=0)
    default_time_in_force: str = Field(default="DAY")
    order_timeout_seconds: int = Field(default=30, ge=1)
    max_order_retries: int = Field(default=3, ge=0)
    enable_smart_routing: bool = Field(default=False)
    
    @field_validator("default_broker")
    @classmethod
    def validate_broker(cls, v: str, info) -> str:
        """Ensure default broker is in valid set."""
        valid = info.data.get("valid_brokers", set())
        if v not in valid:
            raise ValueError(f"Invalid broker: {v}. Must be one of: {valid}")
        return v


class DataConfig(BaseSettings):
    """Data and persistence configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AMATIX_DATA_")
    
    # Database
    database_url: Optional[str] = Field(default=None)
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0")
    
    # Event store
    event_store_path: Path = Field(default=Path("./event_store"))
    journal_enabled: bool = Field(default=True)
    journal_max_events: int = Field(default=100000)
    
    # Market data
    market_data_cache_seconds: int = Field(default=300)
    ohlcv_lookback_days: int = Field(default=365)
    
    @field_validator("event_store_path")
    @classmethod
    def create_event_store_path(cls, v: Path) -> Path:
        """Ensure event store directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class SignalConfig(BaseSettings):
    """Signal generation configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AMATIX_SIGNAL_")
    
    min_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for actionable signals",
    )
    elite_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Elite signal threshold",
    )
    signal_half_life_minutes: int = Field(
        default=20,
        ge=1,
        description="Minutes before signal considered stale",
    )
    news_poll_interval_seconds: int = Field(
        default=300,
        ge=10,
        description="News feed polling interval",
    )
    cooldown_minutes: int = Field(
        default=15,
        ge=0,
        description="Cooldown between signals on same ticker",
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AMATIX_OBS_")
    
    log_level: str = Field(default="INFO")
    log_format: LogFormat = Field(default=LogFormat.JSON)
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    tracing_enabled: bool = Field(default=False)
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    health_check_interval_seconds: int = Field(default=30)
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of: {valid_levels}")
        return v_upper


class Settings(BaseSettings):
    """Global AMATIS settings.
    
    Configuration is loaded from environment variables with the following
    precedence:
        1. Environment variables (highest priority)
        2. .env file
        3. Default values (lowest priority)
    
    Environment variable prefix: AMATIX_
    
    Example:
        AMATIX_ENVIRONMENT=production
        AMATIX_RISK_MAX_DAILY_DRAWDOWN=0.02
        AMATIX_DATABASE_URL=postgresql://...
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra fields for forward compatibility
    )
    
    # Core settings
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    project_name: str = Field(default="amatix")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    
    # Component configs
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    # Feature flags
    enable_ml: bool = Field(default=False)
    enable_news_signals: bool = Field(default=True)
    enable_price_signals: bool = Field(default=False)
    enable_auto_trading: bool = Field(default=False)
    
    @model_validator(mode="after")
    def validate_safety(self) -> Settings:
        """Enforce safety constraints."""
        # Auto-trading only allowed in production if explicitly enabled
        if self.environment == Environment.PRODUCTION:
            if not self.risk.paper_mode and not self.enable_auto_trading:
                # Force paper mode if auto-trading not explicitly enabled
                self.risk.paper_mode = True
        
        return self
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING
    
    def to_dict(self) -> Dict[str, Any]:
        """Export settings to dictionary (for logging/debugging).
        
        Excludes sensitive values like database URLs.
        """
        return {
            "environment": self.environment.value,
            "version": self.version,
            "debug": self.debug,
            "risk": {
                k: v for k, v in self.risk.model_dump().items()
                if k not in {"secrets"}
            },
            "execution": self.execution.model_dump(),
            "signal": self.signal.model_dump(),
            "features": {
                "enable_ml": self.enable_ml,
                "enable_news_signals": self.enable_news_signals,
                "enable_price_signals": self.enable_price_signals,
                "enable_auto_trading": self.enable_auto_trading,
            },
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Settings are loaded once and cached for the process lifetime.
    Use this function to access settings throughout the codebase.
    
    Returns:
        Settings instance
    """
    return Settings()


# Convenience alias
AmatixConfig = Settings
