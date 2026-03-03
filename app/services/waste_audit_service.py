from app.ml.classifier import WasteClassifier
from app.agent.policy import CityPolicyService
from app.agent.prompt_builder import PromptBuilder
from app.agent.llm_client import LLMClient


class WasteAuditService:
    """
    Orchestrates ML classification and AI agent disposal planning.
    """

    def __init__(self):
        self.classifier = WasteClassifier()
        self.llm_client = LLMClient()

    def classify(self, text: str) -> dict:
        """
        Returns classification result:
        {
            "label": str,
            "confidence": float
        }
        """
        return self.classifier.predict(text)

    def generate_disposal_plan(self, text: str) -> dict:
        result = self.classify(text)

        category = result["label"]
        confidence = result["confidence"]

        # Handle uncertain predictions safely
        if category == "Uncertain":
            return {
                "text": text,
                "category": "Uncertain",
                "confidence": confidence,
                "disposal_plan": (
                    "I am not confident about this waste type. "
                    "Please provide more specific details about the item."
                )
            }

        # Logic-based decision (AGENCY)
        policy = CityPolicyService.get_policy(category)

        if category == "Hazardous":
            policy += " Ensure it is kept away from children and pets."
        elif category == "Recyclable":
            policy += " Make sure it is clean and dry."

        prompt = PromptBuilder.build_prompt(
            description=text,
            category=category,
            policy=policy
        )

        llm_response = self.llm_client.generate(prompt)

        if not llm_response:
            llm_response = policy

        return {
            "text": text,
            "category": category,
            "confidence": confidence,
            "disposal_plan": llm_response
        }
