import logging
import requests
from requests.exceptions import Timeout, RequestException

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Handles communication with the Groq LLM API.
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.base_url = settings.LLM_BASE_URL
        self.timeout = settings.LLM_TIMEOUT
        self.temperature = settings.LLM_TEMPERATURE

    def generate(self, prompt: str) -> str | None:
        """
        Send prompt to LLM and return generated text.
        """

        if not self.api_key:
            logger.warning("LLM API key is missing.")
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(
                    "Groq API returned error: %s",
                    response.text
                )
                return None

            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                logger.warning("LLM returned empty response.")
                return None

            # Limit hallucinated long outputs
            if len(content) > 1500:
                content = content[:1500]

            return content

        except Timeout:
            logger.warning("LLM request timed out.")
        except RequestException as e:
            logger.error("LLM request failed: %s", e)
        except Exception:
            logger.exception("Unexpected LLM error.")

        return None
