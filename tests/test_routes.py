def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_classify_endpoint_success(client):
    response = client.post("/classify", json={"text": "banana peel"})
    body = response.json()

    assert response.status_code == 200
    assert body["text"] == "banana peel"
    assert body["category"] == "Compost"
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_endpoint_blank_text_returns_400(client):
    response = client.post("/classify", json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty."


def test_disposal_endpoint_success(client):
    response = client.post("/disposal", json={"text": "banana peel"})
    body = response.json()

    assert response.status_code == 200
    assert body["text"] == "banana peel"
    assert body["category"] == "Compost"
    assert body["confidence"] == 0.91
    assert "compost bin" in body["disposal_plan"].lower()


def test_disposal_endpoint_blank_text_returns_400(client):
    response = client.post("/disposal", json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty."


def test_request_validation_rejects_missing_text(client):
    response = client.post("/classify", json={})
    assert response.status_code == 422


def test_request_validation_rejects_too_long_text(client):
    response = client.post("/classify", json={"text": "a" * 501})
    assert response.status_code == 422
