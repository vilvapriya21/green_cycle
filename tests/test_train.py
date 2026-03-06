from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.ml import train


def test_load_data_reads_valid_dataset(tmp_path):
    data_path = tmp_path / "waste_data.csv"
    df = pd.DataFrame(
        {
            "text": ["banana peel", "plastic bottle", "used battery"],
            "label": ["Compost", "Recyclable", "Hazardous"],
        }
    )
    df.to_csv(data_path, index=False)

    X, y = train.load_data(data_path)

    assert len(X) == 3
    assert len(y) == 3
    assert list(X) == ["banana peel", "plastic bottle", "used battery"]
    assert list(y) == ["Compost", "Recyclable", "Hazardous"]


def test_load_data_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        train.load_data(missing_path)


def test_load_data_raises_for_missing_columns(tmp_path):
    data_path = tmp_path / "bad_data.csv"
    df = pd.DataFrame(
        {
            "description": ["banana peel"],
            "category": ["Compost"],
        }
    )
    df.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="Dataset must contain 'text' and 'label' columns."):
        train.load_data(data_path)


def test_load_data_drops_missing_rows(tmp_path):
    data_path = tmp_path / "partial_data.csv"
    df = pd.DataFrame(
        {
            "text": ["banana peel", None, "used battery"],
            "label": ["Compost", "Recyclable", None],
        }
    )
    df.to_csv(data_path, index=False)

    X, y = train.load_data(data_path)

    assert len(X) == 1
    assert len(y) == 1
    assert list(X) == ["banana peel"]
    assert list(y) == ["Compost"]


def test_build_pipeline_returns_expected_pipeline():
    pipeline = train.build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    assert vectorizer.ngram_range == (1, 1)
    assert vectorizer.max_df == 0.90
    assert vectorizer.min_df == 2
    assert vectorizer.sublinear_tf is True

    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 1000
    assert classifier.solver == "lbfgs"
    assert classifier.random_state == 42


def test_train_model_saves_pipeline(monkeypatch, tmp_path):
    sample_X = pd.Series(
        [
            "banana peel",
            "apple core",
            "plastic bottle",
            "glass jar",
            "used battery",
            "paint can",
            "coffee grounds",
            "cardboard box",
            "light bulb",
            "food scraps",
            "newspaper",
            "vegetable peels",
        ]
    )
    sample_y = pd.Series(
        [
            "Compost",
            "Compost",
            "Recyclable",
            "Recyclable",
            "Hazardous",
            "Hazardous",
            "Compost",
            "Recyclable",
            "Hazardous",
            "Compost",
            "Recyclable",
            "Compost",
        ]
    )

    saved = {}

    class DummyPipeline:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return list(y for y in sample_y.iloc[: len(X)])

    def fake_load_data():
        return sample_X, sample_y

    def fake_build_pipeline():
        return DummyPipeline()

    def fake_train_test_split(X, y, test_size, random_state, stratify):
        return (
            X.iloc[:9],
            X.iloc[9:],
            y.iloc[:9],
            y.iloc[9:],
        )

    def fake_cross_val_score(pipeline, X, y, cv=None, scoring=None):
        return pd.Series([0.90, 0.92, 0.91, 0.89, 0.93])

    def fake_dump(pipeline, path):
        saved["pipeline"] = pipeline
        saved["path"] = path

    monkeypatch.setattr(train, "load_data", fake_load_data)
    monkeypatch.setattr(train, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(train, "train_test_split", fake_train_test_split)
    monkeypatch.setattr(train, "cross_val_score", fake_cross_val_score)
    monkeypatch.setattr(train.joblib, "dump", fake_dump)
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "pipeline.joblib")

    train.train_model()

    assert "pipeline" in saved
    assert saved["path"] == tmp_path / "pipeline.joblib"
