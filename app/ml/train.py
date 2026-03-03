"""
Model Choice Justification:

Logistic Regression is appropriate for this task because:
- TF-IDF creates high-dimensional sparse vectors.
- Logistic Regression performs well on sparse linear data.
- L2 regularization reduces overfitting.
- It is efficient and interpretable.

Overfitting Prevention:
- Stratified 80/20 train-test split.
- Train vs Test accuracy comparison.
- 5-fold stratified cross-validation.
- Regularization (C=0.5).
- Controlled feature space (unigrams only).
"""

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from app.ml.preprocessor import clean_text
from app.config import settings


def load_data():
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(settings.DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    return df["text"], df["label"]


def build_pipeline():
    """
    Build full sklearn pipeline:
    TF-IDF vectorizer + Logistic Regression classifier.
    """

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 1),
        max_df=0.9,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    classifier = LogisticRegression(
        max_iter=1000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs"
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])

    return pipeline


def train_model():
    """
    Train and evaluate model.
    """

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Train Accuracy:", round(train_accuracy, 4))
    print("Test Accuracy:", round(test_accuracy, 4))
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv)

    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", round(cv_scores.mean(), 4))

    # Save full pipeline
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.MODEL_PATH)

    print("Full pipeline saved successfully.")


if __name__ == "__main__":
    train_model()
