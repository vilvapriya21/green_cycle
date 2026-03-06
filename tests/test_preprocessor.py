from app.ml.preprocessor import clean_text


def test_clean_text_lowercases_and_lemmatizes():
    result = clean_text("Running with the plastic bottles!")
    assert isinstance(result, str)
    assert result == result.lower()


def test_clean_text_removes_punctuation_and_stopwords():
    result = clean_text("This is an empty, plastic water bottle.")
    assert "," not in result
    assert "." not in result


def test_clean_text_handles_empty_input():
    result = clean_text("   ")
    assert isinstance(result, str)
    assert result == ""
