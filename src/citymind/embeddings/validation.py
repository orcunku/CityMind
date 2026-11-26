"""CITYMIND semantic embedding input validation."""

from citymind.embeddings import config
from citymind.embeddings.errors import EmbeddingInputError


def validate_embedding_text(text: str) -> str:
    """Validate and normalize embedding input text."""
    if not isinstance(text, str):
        raise EmbeddingInputError("Invalid embedding input type")

    if len(text) > config.MAX_INPUT_CHARS:
        raise EmbeddingInputError("Embedding input too long")

    text = text.strip()

    if not text:
        raise EmbeddingInputError("Empty embedding input")

    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise EmbeddingInputError(
            "Invalid embedding input encoding"
        ) from None

    return text
