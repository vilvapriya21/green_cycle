"""
preprocessor.py
---------------
Text preprocessing module for the Green-Cycle waste classifier.

Handles all NLP cleaning steps:
    - Lowercasing
    - Non-alphabetical character removal
    - Lemmatisation via spaCy
    - Stopword removal
    - Short/meaningless token removal

This same function is used at both training time and inference time
to guarantee consistent feature representation.
"""

import re

import spacy

# Load spaCy model once at module level — avoids reloading on every call.
# Run `python -m spacy download en_core_web_sm` before using this module.
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean and normalise raw text for ML feature extraction.

    Pipeline:
        1. Validate input.
        2. Lowercase entire string.
        3. Strip non-alphabetical characters (keeps spaces).
        4. Run spaCy NLP pipeline for lemmatisation + stopword detection.
        5. Filter out stopwords, punctuation, pronouns, and short tokens.
        6. Return cleaned, space-joined token string.

    Args:
        text (str): Raw waste item description, e.g. "Empty Plastic Bottle!".

    Returns:
        str: Cleaned lemmatised string, e.g. "empty plastic bottle".
             Returns empty string if input is invalid.

    Example:
        >>> clean_text("Used AA Batteries!!!")
        'use battery'
        >>> clean_text("Empty plastic water bottle")
        'empty plastic water bottle'
    """

    # --- Guard: reject non-string or empty input ---
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Normalise case
    text = text.lower()

    # Step 2: Remove anything that is not a letter or space
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Step 3: Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    # Step 4: Pass through spaCy for lemmatisation and linguistic analysis
    doc = nlp(text)

    # Step 5: Filter tokens
    tokens = [
        token.lemma_                        # use base lemma form
        for token in doc
        if not token.is_stop               # drop stopwords (e.g. "the", "is")
        and not token.is_punct             # drop punctuation tokens
        and token.lemma_.strip() != ""     # drop whitespace-only tokens
        and token.lemma_ != "-PRON-"       # drop unresolved pronoun lemmas
        and len(token.lemma_) > 2          # drop very short noise tokens
    ]

    return " ".join(tokens)
