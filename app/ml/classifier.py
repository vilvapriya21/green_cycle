import joblib
from app.config import settings


class WasteClassifier:
    """
    Loads trained sklearn pipeline for inference.
    """

    def __init__(self):
        if not settings.MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found. Train the model first.")

        self.model = joblib.load(settings.MODEL_PATH)

    def predict(self, text: str) -> dict:
        """
        Predict waste category and return confidence.
        """

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]

        confidence = max(probabilities)

        if confidence < 0.55:
            prediction = "Uncertain"

        return {
            "label": prediction,
            "confidence": round(float(confidence), 4)
        }
