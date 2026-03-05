"""
train.py
--------
Model training script for the Green-Cycle waste classifier.

Produces:
    - models/pipeline.joblib (TF-IDF + Logistic Regression pipeline)

Usage:
    python -m app.ml.train
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.config import settings
from app.ml.preprocessor import clean_text

logger = logging.getLogger(__name__)

DATA_PATH: Path = settings.DATA_PATH
MODEL_DIR: Path = settings.MODEL_DIR
MODEL_PATH: Path = settings.MODEL_PATH


def load_data(path: Path = DATA_PATH) -> Tuple[pd.Series, pd.Series]:
    """
    Load dataset from CSV and validate required columns.
    Expected columns: text, label
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    logger.info("Loaded dataset: %d rows", len(df))
    logger.info("Class distribution:\n%s", df["label"].value_counts().to_string())

    return df["text"], df["label"]


def build_pipeline() -> Pipeline:
    """
    Build TF-IDF + Logistic Regression pipeline.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 1),
        max_df=0.90,
        min_df=2,
        sublinear_tf=True,
    )

    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_model() -> None:
    """
    Train, evaluate, and save the ML pipeline.
    """
    logger.info("Starting training...")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info("Split done | train=%d | test=%d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Accuracy
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    logger.info("Accuracy | train=%.4f | test=%.4f", train_acc, test_acc)

    # Classification report
    logger.info("Classification report (test):\n%s", classification_report(y_test, pipeline.predict(X_test)))

    # Confusion matrix (optional but good)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, pipeline.predict(X_test), labels=labels)
    logger.info("Confusion matrix (labels=%s):\n%s", labels, cm)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    logger.info("CV mean=%.4f | std=%.4f", cv_scores.mean(), cv_scores.std())

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Saved pipeline to: %s", MODEL_PATH)

    logger.info("Training complete.")


if __name__ == "__main__":
    train_model()
