from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Green Cycle API",
    description="Waste Classification and Disposal Recommendation System",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Green Cycle API is running"}
