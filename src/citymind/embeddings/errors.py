"""CITYMIND semantic embedding exceptions."""


class EmbeddingError(Exception):
    """Base embedding error."""


class EmbeddingConfigurationError(EmbeddingError):
    """Invalid embedding configuration."""


class EmbeddingModelError(EmbeddingError):
    """Model loading or integrity failure."""


class EmbeddingInputError(EmbeddingError):
    """Invalid embedding input."""


class EmbeddingInferenceError(EmbeddingError):
    """Embedding inference failure."""


class EmbeddingOutputError(EmbeddingError):
    """Invalid embedding output."""
