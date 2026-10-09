from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    app_name: str = 'sonalysis-api'
    model_dir: Path = Path('artifacts')
    model_version: str = 'transaction-mlp-v1'
    device: str = 'cpu'
    postgres_dsn: str | None = None
    redis_url: str | None = None
    kafka_bootstrap: str | None = None
    lakehouse_path: Path = Path('lakehouse/events')
    min_promote_prauc: float = 0.70
    max_promote_brier: float = 0.20
    score_timeout_ms: int = 80

@lru_cache
def get_settings() -> Settings:
    return Settings()
