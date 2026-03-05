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
    MODEL_PATH: Path = MODEL_DIR / "pipeline.joblib"

    # LLM Configuration
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "llama3-8b-8192"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    LLM_TIMEOUT: int = 15
    LLM_TEMPERATURE: float = 0.3

    # City Policies
    CITY_POLICIES: dict = {
        "Recyclable": "Rinse the item and place it in the blue recycling bin.",
        "Compost": "Place the item in the green compost bin.",
        "Hazardous": "Seal the item in a leak-proof container and take it to the Hazardous Waste Facility."
    }

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()