from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Loaded from environment variables or .env file.
    """

    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Data
    DATA_PATH: Path = BASE_DIR / "data" / "waste_data.csv"

    # Models
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = MODEL_DIR / "classifier.joblib"

    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # openai | groq | ollama
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
