from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_classify_endpoint():
    response = client.post(
        "/classify",
        json={"text": "banana peel"}
    )

    assert response.status_code == 200
    assert "category" in response.json()
