import requests

from app.agent.llm_client import LLMClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_generate_returns_none_for_empty_prompt():
    client = LLMClient()
    client.api_key = "test-key"

    assert client.generate("   ") is None


def test_generate_returns_none_when_api_key_missing():
    client = LLMClient()
    client.api_key = None

    assert client.generate("hello") is None


def test_generate_success(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=200,
            payload={
                "choices": [
                    {"message": {"content": "Place it in the green compost bin."}}
                ]
            },
        )

    monkeypatch.setattr("app.agent.llm_client.requests.post", fake_post)

    client = LLMClient()
    client.api_key = "test-key"

    result = client.generate("prompt")
    assert result == "Place it in the green compost bin."


def test_generate_handles_rate_limit(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(status_code=429, text="rate limited")

    monkeypatch.setattr("app.agent.llm_client.requests.post", fake_post)

    client = LLMClient()
    client.api_key = "test-key"

    assert client.generate("prompt") is None


def test_generate_handles_non_200_response(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(status_code=500, text="server error")

    monkeypatch.setattr("app.agent.llm_client.requests.post", fake_post)

    client = LLMClient()
    client.api_key = "test-key"

    assert client.generate("prompt") is None


def test_generate_handles_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr("app.agent.llm_client.requests.post", fake_post)

    client = LLMClient()
    client.api_key = "test-key"

    assert client.generate("prompt") is None
