"""
train.py
--------
Model training script for the Green-Cycle waste classifier.

Produces:
    - models/pipeline.joblib (TF-IDF + Logistic Regression pipeline)

Usage:
    python -m app.ml.train

Model Choice — Logistic Regression:
    TF-IDF vectorisation produces high-dimensional sparse vectors (one dimension
    per vocabulary token). Logistic Regression is well suited to this because:
    - It performs linear classification directly in the sparse feature space.
    - Built-in L2 regularisation (controlled by C) penalises large coefficients,
      preventing the model from over-relying on any single token.
    - It outputs calibrated class probabilities via predict_proba, which are used
      downstream for confidence thresholding.
    KNN was considered but rejected: in high-dimensional sparse space, Euclidean
    distances become unreliable (curse of dimensionality), and KNN has no
    built-in regularisation mechanism.

Overfitting Prevention:
    - Stratified 80/20 train-test split preserves class proportions in both sets.
    - Train accuracy vs test accuracy are compared: a gap under 5% is healthy.
    - 5-fold stratified cross-validation checks that performance is consistent
      across different data splits, not a result of one lucky held-out set.
    - L2 regularisation (C=1.0) keeps model coefficients small.
    - min_df=2 in TF-IDF drops rare tokens that would cause the model to memorise
      one-off training phrases.
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

    TF-IDF parameters:
        preprocessor=clean_text : applies NLP cleaning inside the vectorizer so
            the same transformation runs at both training and inference time.
        ngram_range=(1, 1)      : unigrams only — bigrams on a 301-sample dataset
            produce sparse, noisy features with little benefit.
        max_df=0.90             : drops tokens appearing in >90% of documents;
            these are too generic to discriminate between classes.
        min_df=2                : drops tokens appearing in only one document;
            these are likely typos or noise and would cause overfitting.
        sublinear_tf=True       : applies log(1+tf) scaling to reduce the
            outsized influence of very high-frequency terms.

    Logistic Regression parameters:
        C=1.0                   : inverse regularisation strength; lower values
            apply stronger L2 regularisation. C=1.0 confirmed optimal via
            C-value sweep in the notebook.
        class_weight="balanced" : automatically adjusts class penalties so minor
            class imbalance does not bias predictions toward the majority class.
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
    # Overfitting check: if train_acc >> test_acc (gap > 5%) the model has
    # memorised training examples rather than learning general patterns.
    # A healthy gap is under 5%, confirmed by cross-validation below.
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
    # Stratified 5-fold CV rotates the held-out set across all data so every
    # sample is tested exactly once. A mean CV score close to test_acc confirms
    # the result was not a lucky split. A std below 0.05 confirms the model is
    # stable and not sensitive to which samples land in the test set.
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
