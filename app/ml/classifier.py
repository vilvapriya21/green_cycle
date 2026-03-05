"""
classifier.py
-------------
Inference module for the Green-Cycle waste classifier.

Loads the trained sklearn pipeline from disk and exposes a predict() method
used by the service/API layer.

Design decisions:
    - Pipeline is loaded ONCE at instantiation — not on every request.
    - Returns both label and confidence for transparent API responses.
    - Thresholding is handled in the service layer (business rule),
      so the classifier stays a pure inference component.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: Path = settings.MODEL_PATH


class WasteClassifier:
    """
    Wrapper around the trained TF-IDF + Logistic Regression pipeline.

    Attributes:
        model_path (Path): Path to the saved joblib pipeline file.
        pipeline (Any): Loaded sklearn Pipeline instance.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        """
        Load the trained pipeline from disk.

        Args:
            model_path (Path): Path to the joblib pipeline file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the model exists but cannot be loaded.
        """
        self.model_path = model_path

        if not self.model_path.exists():
            logger.error("Model file not found at: %s", self.model_path)
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                "Run `python -m app.ml.train` to train and save the model first."
            )

        try:
            logger.info("Loading ML pipeline from: %s", self.model_path)
            self.pipeline = joblib.load(self.model_path)
            logger.info("ML pipeline loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load ML pipeline from %s | err=%s", self.model_path, e)
            raise RuntimeError(
                f"Failed to load model pipeline from '{self.model_path}'. "
                "The file may be corrupted or incompatible."
            ) from e

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the waste label and confidence for a given text.

        Note:
            This method does NOT apply any confidence threshold.
            Thresholding is handled by the service layer.

        Args:
            text (str): Raw waste description.

        Returns:
            dict: {"label": str, "confidence": float}

        Raises:
            ValueError: If input text is empty or invalid.
            RuntimeError: If inference fails unexpectedly.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Input text cannot be empty. Please provide a description of the waste item."
            )

        try:
            probabilities = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_

            best_idx = int(probabilities.argmax())
            label = str(classes[best_idx])
            confidence = float(probabilities[best_idx])

            logger.debug(
                "Prediction completed | label=%s | confidence=%.4f | text_len=%d",
                label,
                confidence,
                len(text),
            )

            return {"label": label, "confidence": confidence}

        except Exception as e:
            logger.exception("Model inference failed | text=%r | err=%s", text, e)
            raise RuntimeError("Model inference failed.") from e

    def get_all_probabilities(self, text: str) -> Dict[str, float]:
        """
        Return probabilities for all classes (debug utility).

        Args:
            text (str): Raw waste description.

        Returns:
            dict: {class_name: probability}

        Raises:
            ValueError: If input text is empty.
            RuntimeError: If probability computation fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text cannot be empty.")

        try:
            probabilities = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_

            probs = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

            logger.debug(
                "Probability breakdown | text_len=%d | probs=%s",
                len(text),
                {k: round(v, 4) for k, v in probs.items()},
            )

            return probs

        except Exception as e:
            logger.exception("Failed to compute probabilities | text=%r | err=%s", text, e)
            raise RuntimeError("Probability computation failed.") from e