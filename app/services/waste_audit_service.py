"""
Service layer for orchestrating waste classification and disposal planning.

This module connects:
- ML classification (WasteClassifier)
- City policy retrieval (CityPolicyService)
- Prompt construction (PromptBuilder)
- LLM generation (LLMClient)

It ensures:
- input validation
- confidence gating
- safe fallback behavior when LLM fails or returns unsafe output
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.agent.llm_client import LLMClient
from app.agent.policy import CityPolicyService
from app.agent.prompt_builder import PromptBuilder
from app.config import settings
from app.ml.classifier import WasteClassifier

logger = logging.getLogger(__name__)


class WasteAuditService:
    """
    Orchestrates ML classification and AI-based disposal recommendation generation.

    The service:
    - sanitizes user input
    - predicts a waste category using the ML classifier
    - gates low-confidence predictions
    - retrieves a category-specific city policy
    - generates a disposal plan via LLM with policy-based fallback and safety validation
    """

    # Confidence threshold below which the system responds with "Uncertain"
    MIN_CONFIDENCE: float = settings.MIN_CLASSIFICATION_CONFIDENCE

    # Simple safety filter: if LLM suggests unsafe disposal, revert to policy fallback
    FORBIDDEN_PATTERNS = settings.LLM_FORBIDDEN_PATTERNS

    def __init__(
        self,
        classifier: WasteClassifier | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the service with optional injected dependencies.

        Dependency injection keeps production defaults simple while making the
        service easy to unit test without loading the real model or calling
        external APIs.
        """
        self.classifier = classifier or WasteClassifier()
        self.llm_client = llm_client or LLMClient()

    def _sanitize_input(self, text: str) -> str:
        """
        Normalize user input text.

        Args:
            text (str): raw input

        Returns:
            str: stripped text or empty string if invalid
        """
        if not text or not text.strip():
            return ""
        return text.strip()

    def _is_meaningless(self, text: str) -> bool:
        """
        Detect meaningless / low-signal inputs (e.g., very short or repeated characters).

        Args:
            text (str): sanitized input

        Returns:
            bool: True if input is likely meaningless
        """
        if len(text) < 3:
            return True
        if len(set(text.lower())) <= 2:
            return True
        return False

    def _uncertain_response(self, text: str, confidence: float = 0.0) -> Dict[str, Any]:
        """
        Standard response for uncertain classification or low confidence.

        Args:
            text (str): original text
            confidence (float): model confidence score

        Returns:
            dict: disposal response payload
        """
        return {
            "text": text,
            "category": "Uncertain",
            "confidence": float(confidence),
            "disposal_plan": (
                "I am not confident about this waste type. "
                "Please provide a more specific description."
            ),
        }

    def _validate_llm_response(self, response: Optional[str], policy: str, category: str) -> str:
        """
        Validate the LLM output against basic safety constraints.

        If the response is empty/too short or contains unsafe patterns, fallback to policy.

        Args:
            response (Optional[str]): LLM text response
            policy (str): policy fallback text
            category (str): predicted category

        Returns:
            str: safe disposal plan
        """
        if not response or len(response.strip()) < 20:
            return policy

        lower_resp = response.lower()

        for phrase in self.FORBIDDEN_PATTERNS:
            if phrase in lower_resp:
                logger.warning("Unsafe LLM phrase detected (%r). Falling back to policy.", phrase)
                return policy

        # Category-specific guardrails
        if category == "Hazardous" and "recycle" in lower_resp:
            logger.warning("LLM suggested recycling for Hazardous waste. Falling back to policy.")
            return policy

        return response.strip()

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify the waste description using the ML classifier.

        IMPORTANT:
        - Classifier returns raw label + confidence
        - This service applies confidence gating (MIN_CONFIDENCE)

        Args:
            text (str): waste description

        Returns:
            dict: {"label": str, "confidence": float}
        """
        clean_text = self._sanitize_input(text)

        if not clean_text or self._is_meaningless(clean_text):
            logger.info("Classification skipped: empty/meaningless input | text=%r", text)
            return {"label": "Uncertain", "confidence": 0.0}

        try:
            pred = self.classifier.predict(clean_text)

            label = pred.get("label", "Uncertain")
            confidence = float(pred.get("confidence", 0.0))

            # Apply confidence gating here (single source of truth)
            if confidence < self.MIN_CONFIDENCE:
                logger.info(
                    "Confidence below threshold | raw_label=%s | confidence=%.3f | threshold=%.2f",
                    label,
                    confidence,
                    self.MIN_CONFIDENCE,
                )
                return {"label": "Uncertain", "confidence": confidence}

            logger.info(
                "Classification success | label=%s | confidence=%.3f",
                label,
                confidence,
            )
            return {"label": label, "confidence": confidence}

        except Exception as e:
            # Never swallow errors silently; log for debugging
            logger.exception("Classification failed | text=%r | err=%s", clean_text, e)
            return {"label": "Uncertain", "confidence": 0.0}

    def generate_disposal_plan(self, text: str) -> Dict[str, Any]:
        """
        Generate disposal guidance by combining:
        - ML classification
        - city policy retrieval
        - LLM completion (Groq/OpenAI-compatible endpoint)
        - safety validation and fallback

        Args:
            text (str): waste description

        Returns:
            dict: {"text": str, "category": str, "confidence": float, "disposal_plan": str}
        """
        clean_text = self._sanitize_input(text)

        if not clean_text:
            logger.info("Disposal plan skipped: empty input")
            return self._uncertain_response(text)

        if len(clean_text) > 500:
            logger.warning("Input truncated to 500 chars | original_len=%d", len(clean_text))
            clean_text = clean_text[:500]

        result = self.classify(clean_text)
        category = result.get("label", "Uncertain")
        confidence = float(result.get("confidence", 0.0))

        if category == "Uncertain" or confidence < self.MIN_CONFIDENCE:
            logger.info(
                "Low confidence classification | category=%s | confidence=%.3f",
                category,
                confidence,
            )
            return self._uncertain_response(text, confidence)

        # Policy retrieval
        try:
            policy = CityPolicyService.get_policy(category)
        except Exception as e:
            logger.exception("Policy retrieval failed | category=%s | err=%s", category, e)
            policy = (
                "Follow your local municipal guidelines for this waste type. "
                "If unsure, contact your local waste management authority."
            )

        # Add category-specific safety notes
        if category == "Hazardous":
            policy += " Ensure it is kept away from children and pets."
        elif category == "Recyclable":
            policy += " Make sure it is clean and dry before recycling."

        # Prompt building
        try:
            prompt = PromptBuilder.build_prompt(
                description=clean_text,
                category=category,
                policy=policy,
            )
        except Exception as e:
            logger.exception("Prompt building failed | err=%s", e)
            # If prompt building fails, fallback immediately to policy
            return {
                "text": text,
                "category": category,
                "confidence": confidence,
                "disposal_plan": policy,
            }

        # LLM call
        llm_response = self.llm_client.generate(prompt)
        if llm_response is None:
            logger.warning("LLM returned None. Falling back to policy | category=%s", category)
        else:
            logger.info("LLM responded | chars=%d | category=%s", len(llm_response), category)

        safe_response = self._validate_llm_response(llm_response, policy, category)

        return {
            "text": text,
            "category": category,
            "confidence": confidence,
            "disposal_plan": safe_response,
        }
