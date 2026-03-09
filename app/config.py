from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.

    Values are loaded from environment variables and optionally from a `.env` file.
    """

    # ✅ pydantic-settings v2 way (reliable)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Data
    DATA_PATH: Path = BASE_DIR / "data" / "waste_data.csv"

    # Models
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = MODEL_DIR / "pipeline.joblib"

    # LLM Configuration (Groq OpenAI-compatible endpoint)
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    LLM_TIMEOUT: int = 20
    LLM_TEMPERATURE: float = 0.2

    # Classification safety threshold
    MIN_CLASSIFICATION_CONFIDENCE: float = 0.55

    # LLM Safety guardrails
    LLM_FORBIDDEN_PATTERNS: list[str] = [
        "burn the battery",
        "throw in river",
        "mix with food waste",
    ]

    # City Policies (simulated)
    CITY_POLICIES: dict = {
        "Recyclable": "Rinse the item and place it in the blue recycling bin.",
        "Compost": "Place the item in the green compost bin.",
        "Hazardous": "Seal the item in a leak-proof container and take it to the Hazardous Waste Facility.",
    }


settings = Settings()
