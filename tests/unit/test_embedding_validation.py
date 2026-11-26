import pytest

from citymind.embeddings.config import MAX_INPUT_CHARS
from citymind.embeddings.errors import EmbeddingInputError
from citymind.embeddings.validation import validate_embedding_text


@pytest.mark.parametrize("value", [
    None, True, 123, 1.5, [], {}, b"text"
])
def test_invalid_input_type(value):
    with pytest.raises(EmbeddingInputError):
        validate_embedding_text(value)


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_empty_input(value):
    with pytest.raises(EmbeddingInputError):
        validate_embedding_text(value)


def test_oversized_input():
    with pytest.raises(EmbeddingInputError):
        validate_embedding_text("a" * (MAX_INPUT_CHARS + 1))


def test_maximum_input_length():
    text = "a" * MAX_INPUT_CHARS
    assert validate_embedding_text(text) == text


def test_invalid_unicode():
    with pytest.raises(EmbeddingInputError):
        validate_embedding_text("\ud800")


def test_turkish_input():
    text = "İzmir'de hava kirliliği"
    assert validate_embedding_text(text) == text


def test_whitespace_normalization():
    assert validate_embedding_text(
        "  Air quality monitoring  "
    ) == "Air quality monitoring"
