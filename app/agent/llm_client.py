import requests
import logging
from requests.exceptions import Timeout, RequestException
from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Handles communication with Groq LLM.
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.timeout = 15

    def generate(self, prompt: str) -> str | None:
        """
        Sends prompt to Groq.
        Returns generated text or None if failure occurs.
        """

        if not self.api_key:
            logger.warning("LLM API key missing.")
            return None

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(f"Groq error: {response.text}")
                return None

            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                return None

            if len(content) > 1500:  # avoid excessive hallucinated output
                content = content[:1500]

            return content

        except Timeout:
            logger.warning("LLM request timed out.")
        except RequestException as e:
            logger.error(f"LLM request failed: {e}")
        except Exception:
            logger.exception("Unexpected LLM failure.")

        return None
