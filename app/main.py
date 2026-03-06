import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Application factory for Green-Cycle API.
    """
    app = FastAPI(
        title="Green Cycle API",
        description="Waste Classification and Disposal Recommendation System",
        version="1.0.0",
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception occurred.")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    app.include_router(router)

    @app.get("/")
    def root() -> dict:
        return {"message": "Green Cycle API is running"}

    return app


app = create_app()
