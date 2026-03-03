import requests
from app.config import settings


class LLMClient:
    """
    Handles communication with external LLM providers.
    Supports OpenAI, Groq, and Ollama via configurable base URL.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL

    def generate(self, prompt: str) -> str | None:
        """
        Sends prompt to configured LLM provider.
        Returns generated text or None if failure occurs.
        """

        if self.provider in ["openai", "groq"]:
            return self._call_openai_compatible(prompt)

        elif self.provider == "ollama":
            return self._call_ollama(prompt)

        return None

    def _call_openai_compatible(self, prompt: str) -> str | None:
        """
        OpenAI and Groq both support OpenAI-compatible API format.
        """

        if not self.api_key or not self.base_url:
            return None

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=15,
            )

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.RequestException:
            return None

    def _call_ollama(self, prompt: str) -> str | None:
        """
        Calls local Ollama server.
        """

        if not self.base_url:
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=15,
            )

            response.raise_for_status()
            return response.json().get("response")

        except requests.RequestException:
            return None
