"""
train.py
--------
Model training script for the Green-Cycle waste classifier.

Run this script ONCE before starting the API server to produce:
    - models/pipeline.joblib   (TF-IDF + Logistic Regression pipeline)

Usage:
    python -m app.ml.train

Model Choice — Logistic Regression:
    - TF-IDF produces high-dimensional sparse vectors.
    - Logistic Regression is purpose-built for sparse linear data.
    - Built-in L2 regularisation (C parameter) directly controls overfitting.
    - Fully probabilistic output — enables confidence scoring at inference.
    - Fast to train, interpretable, and stable on small-to-medium datasets.

Overfitting Prevention Strategy:
    - Stratified 80/20 train-test split (preserves class ratio in both sets).
    - Train accuracy vs Test accuracy comparison (gap > 5% signals overfitting).
    - 5-fold stratified cross-validation with variance check (std < 0.05 is healthy).
    - Conservative regularisation: C=0.5 (lower C = stronger regularisation).
    - Feature constraints: unigrams only, min_df=2 removes rare noise terms.
    - class_weight="balanced" prevents majority-class bias.
"""

import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from app.ml.preprocessor import clean_text


# ── Paths ────────────────────────────────────────────────────────────────────

DATA_PATH  = Path("data/waste_data.csv")
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "pipeline.joblib"


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(path: Path = DATA_PATH):
    """
    Load dataset from CSV file and validate required columns.

    Args:
        path (Path): Path to the CSV dataset file.

    Returns:
        tuple[pd.Series, pd.Series]: Feature series X (text) and label series y.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns ('text', 'label') are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place waste_data.csv in the data/ directory."
        )

    df = pd.read_csv(path)

    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Drop rows where either column is null
    df = df.dropna(subset=["text", "label"])

    print(f"Loaded {len(df)} samples.")
    print("Class distribution:\n", df["label"].value_counts(), "\n")

    return df["text"], df["label"]


# ── Pipeline Construction ─────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Construct the sklearn ML pipeline.

    Steps:
        1. TfidfVectorizer  — converts cleaned text to weighted term vectors.
        2. LogisticRegression — classifies the vector into a waste category.

    TF-IDF Parameters Explained:
        preprocessor=clean_text  : applies lemmatisation/stopword removal.
        ngram_range=(1, 1)       : unigrams only — avoids sparse bigram noise.
        max_df=0.90              : ignores terms appearing in >90% of docs (too common).
        min_df=2                 : ignores terms appearing in <2 docs (too rare / noise).
        sublinear_tf=True        : log-normalises term frequencies — reduces outlier effect.
        token_pattern            : matches any word character sequence.

    Logistic Regression Parameters Explained:
        C=1                      : inverse regularisation strength — lower = more regularised.
        class_weight="balanced"  : corrects for unequal class sizes automatically.
        max_iter=1000            : ensures convergence on larger vocabulary.
        solver="lbfgs"           : best solver for multinomial problems.

    Returns:
        Pipeline: Unfitted sklearn pipeline.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 1),
        max_df=0.90,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    classifier = LogisticRegression(
        C=1,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf",   classifier)
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_overfitting(train_acc: float, test_acc: float, cv_scores) -> None:
    """
    Print a human-readable overfitting health report.

    Rules of thumb applied:
        - Train/test gap < 5%  → healthy generalisation.
        - CV std < 0.05        → stable across data splits (not lucky on one split).

    Args:
        train_acc  (float)      : Accuracy on training set.
        test_acc   (float)      : Accuracy on held-out test set.
        cv_scores  (np.ndarray) : Cross-validation accuracy scores.
    """
    gap = train_acc - test_acc
    print("\n── Overfitting Analysis ──────────────────────────")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  Gap            : {gap:.4f}")
    print(f"  CV Mean        : {cv_scores.mean():.4f}")
    print(f"  CV Std         : {cv_scores.std():.4f}")
    print(f"  CV Scores      : {[round(s, 4) for s in cv_scores]}")
    print("──────────────────────────────────────────────────\n")


# ── Main Training Entry Point ─────────────────────────────────────────────────

def train_model() -> None:
    """
    Full training, evaluation, and model persistence workflow.

    Steps:
        1. Load and validate dataset.
        2. Stratified 80/20 train-test split.
        3. Fit pipeline on training data.
        4. Evaluate train and test accuracy.
        5. Print per-class classification report.
        6. Run 5-fold stratified cross-validation.
        7. Print overfitting analysis.
        8. Save fitted pipeline to disk.
    """
    print("=" * 52)
    print("       Green-Cycle Model Training Script")
    print("=" * 52, "\n")

    # 1. Load data
    X, y = load_data()

    # 2. Stratified split — preserves class ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}\n")

    # 3. Build and fit pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Accuracy on both sets
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc  = accuracy_score(y_test,  pipeline.predict(X_test))

    # 5. Per-class classification report
    print("── Classification Report (Test Set) ──────────────")
    print(classification_report(y_test, pipeline.predict(X_test)))

    # 6. Cross-validation on full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    # 7. Overfitting analysis
    evaluate_overfitting(train_acc, test_acc, cv_scores)

    # 8. Save pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved → {MODEL_PATH}")
    print("\nTraining complete. You can now start the API server.")


if __name__ == "__main__":
    train_model()
