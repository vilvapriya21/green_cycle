"""
preprocessor.py
---------------
Text preprocessing module for the Green-Cycle waste classifier.

Handles NLP cleaning steps:
    - Lowercasing
    - Non-alphabetical character removal
    - Lemmatisation via spaCy
    - Stopword removal
    - Short/meaningless token removal

The same function is used during both training and inference
to guarantee consistent feature representation.
"""

from __future__ import annotations

import logging
import re

import spacy

logger = logging.getLogger(__name__)

# ── Load spaCy model once (avoid repeated loading per request) ─────────────────

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception:
    logger.warning(
        "spaCy model 'en_core_web_sm' not found. "
        "Falling back to blank English pipeline. "
        "Run: python -m spacy download en_core_web_sm"
    )
    nlp = spacy.blank("en")


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text for ML feature extraction.

    Processing pipeline:
        1. Validate input
        2. Lowercase text
        3. Remove non-alphabetical characters
        4. Collapse whitespace
        5. Apply spaCy lemmatization
        6. Remove stopwords and short tokens

    Args:
        text (str): Raw waste description.

    Returns:
        str: Cleaned token string suitable for TF-IDF.
             Returns empty string if input is invalid.
    """

    # ── Guard: invalid input ────────────────────────────────────────────────
    if not isinstance(text, str) or not text.strip():
        return ""

    # Defensive: avoid extremely long inputs
    if len(text) > 1000:
        logger.debug("Input text truncated during preprocessing (len=%d)", len(text))
        text = text[:1000]

    try:
        # Step 1: lowercase
        text = text.lower()

        # Step 2: remove non-letters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Step 3: collapse spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Step 4: spaCy processing
        doc = nlp(text)

        # Step 5: token filtering
        tokens = []
        for token in doc:
            lemma = token.lemma_.strip()

            if (
                not token.is_stop
                and not token.is_punct
                and lemma
                and lemma != "-PRON-"
                and len(lemma) > 2
            ):
                tokens.append(lemma)

        return " ".join(tokens)

    except Exception as e:
        logger.exception("Text preprocessing failed | text=%r | err=%s", text, e)
        return ""
