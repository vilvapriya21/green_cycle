import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create app FIRST
app = FastAPI(
    title="Green Cycle API",
    description="Waste Classification and Disposal Recommendation System",
    version="1.0.0"
)

# Exception handler AFTER app creation
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred.")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )

# Include routes
app.include_router(router)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Green Cycle API is running"}
