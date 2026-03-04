from app.ml.classifier import WasteClassifier


def test_classifier_prediction():
    clf = WasteClassifier()
    result = clf.predict("empty plastic bottle")

    assert result["label"] in [
        "Recyclable",
        "Compost",
        "Hazardous",
        "Uncertain",
    ]
    assert 0.0 <= result["confidence"] <= 1.0
