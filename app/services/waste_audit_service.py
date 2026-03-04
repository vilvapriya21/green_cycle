from app.ml.classifier import WasteClassifier
from app.agent.policy import CityPolicyService
from app.agent.prompt_builder import PromptBuilder
from app.agent.llm_client import LLMClient


class WasteAuditService:
    """
    Orchestrates ML classification and AI agent disposal planning.
    Adds strong edge-case handling and hallucination safeguards.
    """

    def __init__(self):
        self.classifier = WasteClassifier()
        self.llm_client = LLMClient()

    # --------------------------
    # Utility Safety Functions
    # --------------------------

    def _sanitize_input(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        return text.strip()

    def _is_meaningless(self, text: str) -> bool:
        # Detect gibberish or extremely short input
        if len(text) < 3:
            return True

        unique_chars = len(set(text.lower()))
        if unique_chars <= 2:
            return True

        return False

    def _validate_llm_response(self, response: str, policy: str, category: str) -> str:
        """
        Prevent hallucination:
        - Response must exist
        - Must not contradict category
        - Must not suggest unsafe disposal
        """

        if not response or len(response.strip()) < 20:
            return policy

        lower_resp = response.lower()

        # Basic hallucination keyword checks
        forbidden_patterns = [
            "burn the battery",
            "throw in river",
            "mix with food waste",
        ]

        for phrase in forbidden_patterns:
            if phrase in lower_resp:
                return policy

        # Ensure category consistency
        if category == "Hazardous" and "recycle" in lower_resp:
            return policy

        return response.strip()

    # --------------------------
    # Core Methods
    # --------------------------

    def classify(self, text: str) -> dict:
        """
        Returns classification result:
        {
            "label": str,
            "confidence": float
        }
        """

        clean_text = self._sanitize_input(text)

        if not clean_text or self._is_meaningless(clean_text):
            return {
                "label": "Uncertain",
                "confidence": 0.0
            }

        try:
            return self.classifier.predict(clean_text)
        except Exception:
            return {
                "label": "Uncertain",
                "confidence": 0.0
            }

    def generate_disposal_plan(self, text: str) -> dict:

        clean_text = self._sanitize_input(text)

        if not clean_text:
            return {
                "text": text,
                "category": "Uncertain",
                "confidence": 0.0,
                "disposal_plan": "Text cannot be empty. Please describe the waste item clearly."
            }

        # Limit very long input (token safety)
        if len(clean_text) > 500:
            clean_text = clean_text[:500]

        result = self.classify(clean_text)

        category = result["label"]
        confidence = result["confidence"]

        # Low confidence safety
        if confidence < 0.55:
            category = "Uncertain"

        # If uncertain → do not call LLM
        if category == "Uncertain":
            return {
                "text": text,
                "category": "Uncertain",
                "confidence": confidence,
                "disposal_plan": (
                    "I am not confident about this waste type. "
                    "Please provide a more specific description."
                )
            }

        # --------------------------
        # Logic-based policy decision
        # --------------------------

        policy = CityPolicyService.get_policy(category)

        if category == "Hazardous":
            policy += " Ensure it is kept away from children and pets."
        elif category == "Recyclable":
            policy += " Make sure it is clean and dry before recycling."

        prompt = PromptBuilder.build_prompt(
            description=clean_text,
            category=category,
            policy=policy
        )

        llm_response = self.llm_client.generate(prompt)

        # Validate LLM output against hallucination
        safe_response = self._validate_llm_response(
            llm_response,
            policy,
            category
        )

        return {
            "text": text,
            "category": category,
            "confidence": confidence,
            "disposal_plan": safe_response
        }
