from pathlib import Path

import numpy as np
import pytest

from app.ml.classifier import WasteClassifier


class FakePipeline:
    classes_ = np.array(["Compost", "Hazardous", "Recyclable"])

    def predict_proba(self, texts):
        return np.array([[0.70, 0.10, 0.20]])


def test_classifier_prediction(monkeypatch, tmp_path):
    model_path = tmp_path / "pipeline.joblib"
    model_path.write_text("dummy")

    monkeypatch.setattr("app.ml.classifier.joblib.load", lambda _: FakePipeline())

    clf = WasteClassifier(model_path=model_path)
    result = clf.predict("banana peel")

    assert result["label"] == "Compost"
    assert 0.0 <= result["confidence"] <= 1.0


def test_classifier_rejects_empty_input(monkeypatch, tmp_path):
    model_path = tmp_path / "pipeline.joblib"
    model_path.write_text("dummy")

    monkeypatch.setattr("app.ml.classifier.joblib.load", lambda _: FakePipeline())

    clf = WasteClassifier(model_path=model_path)

    with pytest.raises(ValueError):
        clf.predict("   ")


def test_classifier_missing_model_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        WasteClassifier(model_path=missing_path)
