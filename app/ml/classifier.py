"""
classifier.py
-------------
Inference module for the Green-Cycle waste classifier.

Loads the trained sklearn pipeline from disk and exposes a single
predict() method used by the FastAPI route handlers.

Design decisions:
    - Pipeline is loaded ONCE at instantiation — not on every request.
    - Confidence threshold (0.55) filters low-certainty predictions.
    - Returns both label and confidence for transparent API responses.
    - All edge cases (empty input, missing model) raise typed exceptions
      so the API layer can return appropriate HTTP status codes.
"""

import joblib
from pathlib import Path


# Default model location — matches what train.py writes
DEFAULT_MODEL_PATH = Path("models/pipeline.joblib")

# Predictions below this confidence are returned as "Uncertain"
CONFIDENCE_THRESHOLD = 0.55


class WasteClassifier:
    """
    Wrapper around the trained TF-IDF + Logistic Regression pipeline.

    Attributes:
        model_path (Path)  : Path to the saved joblib pipeline file.
        pipeline   (object): Loaded sklearn Pipeline instance.

    Usage:
        classifier = WasteClassifier()
        result = classifier.predict("empty plastic bottle")
        # → {"label": "Recyclable", "confidence": 0.91}
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        """
        Load the trained pipeline from disk.

        Args:
            model_path (Path): Path to the joblib pipeline file.

        Raises:
            FileNotFoundError: If the model file does not exist at the given path.
                               Instructs user to run the training script first.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run `python -m app.ml.train` to train and save the model first."
            )

        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        """
        Classify a waste item description into a waste category.

        The raw text is passed directly into the pipeline — TF-IDF's
        preprocessor (clean_text) handles all NLP cleaning internally,
        which guarantees the same preprocessing used during training.

        Confidence thresholding:
            If the highest class probability is below CONFIDENCE_THRESHOLD,
            the label is returned as "Uncertain" to avoid overconfident
            wrong predictions on out-of-vocabulary descriptions.

        Args:
            text (str): Raw waste item description from the user.
                        e.g. "Old AA batteries from a remote control"

        Returns:
            dict: {
                "label"      : str   — predicted class or "Uncertain",
                "confidence" : float — probability of the predicted class (0–1)
            }

        Raises:
            ValueError: If the input text is empty or whitespace-only.

        Example:
            >>> clf = WasteClassifier()
            >>> clf.predict("banana peel")
            {"label": "Compost", "confidence": 0.87}
            >>> clf.predict("broken fluorescent tube")
            {"label": "Hazardous", "confidence": 0.93}
        """
        # Guard: reject empty or whitespace-only input
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Input text cannot be empty. "
                "Please provide a description of the waste item."
            )

        # Get predicted label
        predicted_label = self.pipeline.predict([text])[0]

        # Get class probabilities for confidence scoring
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = float(max(probabilities))

        # Apply confidence threshold — flag low-certainty predictions
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Uncertain"

        return {
            "label":      predicted_label,
            "confidence": round(confidence, 4)
        }

    def get_all_probabilities(self, text: str) -> dict:
        """
        Return probabilities for ALL classes — useful for debugging
        and for building confidence breakdowns in API responses.

        Args:
            text (str): Raw waste item description.

        Returns:
            dict: {class_name: probability} for each known class.

        Example:
            >>> clf.get_all_probabilities("empty plastic bottle")
            {"Compost": 0.04, "Hazardous": 0.05, "Recyclable": 0.91}
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        classes       = self.pipeline.classes_
        probabilities = self.pipeline.predict_proba([text])[0]

        return {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
