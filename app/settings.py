import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH, extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = ""
    adk_model: str = "gemini-1.5-flash"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10


settings = Settings()
