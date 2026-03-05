import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging_config import setup_logging

from pathlib import Path
from app.config import settings
from app.ml.train import train_model

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI instance.
    """

    app = FastAPI(
        title="Green Cycle API",
        description="Waste Classification and Disposal Recommendation System",
        version="1.0.0",
    )
    

    @app.on_event("startup")
    def ensure_model_exists():
        if not Path(settings.MODEL_PATH).exists():
            logger.warning("Model not found. Training model now...")
            train_model()
            logger.info("Model training complete.")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception occurred.")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    app.include_router(router)

    @app.get("/")
    def root():
        return {"message": "Green Cycle API is running"}

    return app


app = create_app()