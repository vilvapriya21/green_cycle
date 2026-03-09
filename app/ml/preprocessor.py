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

Why lemmatisation over stemming:
    Stemming applies crude suffix-cutting rules and produces non-words
    (e.g. "batteries" → "batteri"). Lemmatisation uses spaCy's vocabulary
    and morphological analysis to produce real base forms
    (e.g. "batteries" → "battery", "broken" → "break"). Real base forms
    are more meaningful features for a domain where specific item names
    (battery, chemical, peel) carry the primary classification signal.

Why this function is passed into TfidfVectorizer as preprocessor:
    Passing clean_text as the TF-IDF preprocessor ensures the identical
    NLP pipeline runs at training time and inference time. If preprocessing
    were applied separately, any inconsistency between the two stages would
    cause train/serve skew and degrade real-world accuracy.
"""

from __future__ import annotations

import logging
import re

import spacy

logger = logging.getLogger(__name__)

#Load spaCy model once (avoid repeated loading per request)

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
        # Step 1: lowercase — "Paint" and "paint" must map to the same token
        text = text.lower()

        # Step 2: remove non-letters — punctuation and numbers carry no class
        # signal for waste descriptions and add noise to the vocabulary
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Step 3: collapse spaces — normalise any spacing artefacts left after
        # character removal
        text = re.sub(r"\s+", " ", text).strip()

        # Step 4: spaCy processing — runs the full linguistic pipeline to
        # produce lemmas and detect stopword/punctuation flags
        doc = nlp(text)

        # Step 5: token filtering
        # - is_stop: stopwords ("the", "a", "is") appear in all classes and
        #   contribute zero discriminating signal; removing them reduces noise
        # - len(lemma) > 2: very short tokens are typically noise or
        #   prepositions not caught by the stopword list
        # - lemma != "-PRON-": unresolved pronoun placeholders from older spaCy
        #   models are meaningless features
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
