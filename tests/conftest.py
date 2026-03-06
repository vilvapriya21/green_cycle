import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_waste_audit_service
from app.main import app


class StubService:
    def classify(self, text: str) -> dict:
        return {"label": "Compost", "confidence": 0.91}

    def generate_disposal_plan(self, text: str) -> dict:
        return {
            "text": text,
            "category": "Compost",
            "confidence": 0.91,
            "disposal_plan": "Place the item in the green compost bin.",
        }


@pytest.fixture
def client():
    app.dependency_overrides[get_waste_audit_service] = lambda: StubService()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()