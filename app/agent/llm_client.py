"""
llm_client.py
-------------
Simple client for calling an OpenAI-compatible chat completion endpoint.

Default usage:
- Groq API (OpenAI-compatible) via LLM_BASE_URL
"""

import logging
from typing import Optional

import requests
from requests.exceptions import Timeout, RequestException

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Handles communication with the configured LLM API.

    Returns None on failure so the service layer can fall back to policy text.
    """

    def __init__(self) -> None:
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.base_url = settings.LLM_BASE_URL
        self.timeout = settings.LLM_TIMEOUT
        self.temperature = settings.LLM_TEMPERATURE

    def generate(self, prompt: str, max_chars: int = 1500) -> Optional[str]:
        """
        Send a prompt to the LLM and return the generated text.

        Args:
            prompt (str): prompt text
            max_chars (int): cap on returned text length

        Returns:
            Optional[str]: generated text or None if failed
        """
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("LLM called with empty prompt.")
            return None

        if not self.api_key:
            logger.warning("LLM_API_KEY is missing. Skipping LLM call.")
            return None

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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

            if response.status_code == 429:
                logger.warning("LLM rate limited (429). Falling back to policy.")
                return None

            if response.status_code != 200:
                logger.error(
                    "LLM API error | status=%s | body=%s",
                    response.status_code,
                    response.text[:300],
                )
                return None

            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if not isinstance(content, str) or not content.strip():
                logger.warning("LLM returned empty response.")
                return None

            content = content.strip()

            if len(content) > max_chars:
                content = content[:max_chars]

            return content

        except Timeout:
            logger.warning("LLM request timed out.")
            return None
        except RequestException as e:
            logger.error("LLM request failed: %s", e)
            return None
        except Exception:
            logger.exception("Unexpected LLM error.")
            return None
