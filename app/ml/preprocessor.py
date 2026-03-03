import re
import spacy

# Load spaCy English model once at module level for efficiency
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Perform NLP preprocessing on input text.

    Steps:
    - Convert to lowercase
    - Remove non-alphabetical characters
    - Lemmatize tokens
    - Remove stopwords and punctuation
    - Remove short or meaningless tokens

    This function is used for both training and inference
    to ensure consistent text processing.
    """

    if not text or not isinstance(text, str):
        return ""

    # Normalize case
    text = text.lower()

    # Remove non-alphabetical characters (retain spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Process text using spaCy pipeline
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop              # remove stopwords
        and not token.is_punct            # remove punctuation
        and token.lemma_.strip() != ""    # remove empty tokens
        and token.lemma_ != "-PRON-"      # remove unresolved pronoun lemma
        and len(token.lemma_) > 2         # remove very short tokens
    ]

    return " ".join(tokens)
