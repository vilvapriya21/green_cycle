from app.ml.classifier import WasteClassifier


class WasteService:
    """
    Orchestrates business logic between API layer and ML/Agent layers.
    Keeps transport layer independent from business logic.
    """

    def __init__(self):
        self.classifier = WasteClassifier()

    def classify(self, text: str) -> str:
        """
        Classify a waste item description.
        """
        return self.classifier.predict(text)
