from app.ml.classifier import WasteClassifier
from app.agent.policy import CityPolicyService
from app.agent.prompt_builder import PromptBuilder
from app.agent.llm_client import LLMClient


class WasteAuditService:
    """
    Service layer that orchestrates ML classification and AI-based
    disposal recommendation generation.
    """

    MIN_CONFIDENCE = 0.55

    FORBIDDEN_PATTERNS = [
        "burn the battery",
        "throw in river",
        "mix with food waste",
    ]

    def __init__(self):
        self.classifier = WasteClassifier()
        self.llm_client = LLMClient()

    def _sanitize_input(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        return text.strip()

    def _is_meaningless(self, text: str) -> bool:
        if len(text) < 3:
            return True

        if len(set(text.lower())) <= 2:
            return True

        return False

    def _uncertain_response(self, text: str, confidence: float = 0.0):
        return {
            "text": text,
            "category": "Uncertain",
            "confidence": confidence,
            "disposal_plan": (
                "I am not confident about this waste type. "
                "Please provide a more specific description."
            ),
        }

    def _validate_llm_response(self, response: str, policy: str, category: str) -> str:

        if not response or len(response.strip()) < 20:
            return policy

        lower_resp = response.lower()

        for phrase in self.FORBIDDEN_PATTERNS:
            if phrase in lower_resp:
                return policy

        if category == "Hazardous" and "recycle" in lower_resp:
            return policy

        return response.strip()

    def classify(self, text: str) -> dict:

        clean_text = self._sanitize_input(text)

        if not clean_text or self._is_meaningless(clean_text):
            return {"label": "Uncertain", "confidence": 0.0}

        try:
            return self.classifier.predict(clean_text)
        except Exception:
            return {"label": "Uncertain", "confidence": 0.0}

    def generate_disposal_plan(self, text: str) -> dict:

        clean_text = self._sanitize_input(text)

        if not clean_text:
            return self._uncertain_response(text)

        if len(clean_text) > 500:
            clean_text = clean_text[:500]

        result = self.classify(clean_text)

        category = result["label"]
        confidence = result["confidence"]

        if confidence < self.MIN_CONFIDENCE:
            return self._uncertain_response(text, confidence)

        policy = CityPolicyService.get_policy(category)

        if category == "Hazardous":
            policy += " Ensure it is kept away from children and pets."
        elif category == "Recyclable":
            policy += " Make sure it is clean and dry before recycling."

        prompt = PromptBuilder.build_prompt(
            description=clean_text,
            category=category,
            policy=policy,
        )

        llm_response = self.llm_client.generate(prompt)

        safe_response = self._validate_llm_response(
            llm_response,
            policy,
            category,
        )

        return {
            "text": text,
            "category": category,
            "confidence": confidence,
            "disposal_plan": safe_response,
        }
