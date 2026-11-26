"""CITYMIND semantic embedding pooling and normalization."""

import numpy as np

from citymind.embeddings import config
from citymind.embeddings.errors import EmbeddingOutputError


def pool_and_normalize(
    hidden_state: np.ndarray,
    attention_mask: np.ndarray,
) -> list[float]:
    """Generate a normalized semantic embedding."""
    if not isinstance(hidden_state, np.ndarray):
        raise EmbeddingOutputError("Invalid hidden state type")

    if (
        hidden_state.ndim != 3
        or hidden_state.shape[0] != 1
        or hidden_state.shape[2] != config.EMBEDDING_DIMENSION
        or hidden_state.dtype != np.float32
    ):
        raise EmbeddingOutputError("Invalid hidden state tensor")

    if not np.isfinite(hidden_state).all():
        raise EmbeddingOutputError("Non-finite hidden state")

    if (
        not isinstance(attention_mask, np.ndarray)
        or attention_mask.shape != hidden_state.shape[:2]
        or attention_mask.dtype != np.int64
    ):
        raise EmbeddingOutputError("Invalid attention mask")

    if not np.isin(attention_mask, [0, 1]).all():
        raise EmbeddingOutputError("Invalid attention mask values")

    token_count = int(attention_mask.sum())

    if token_count <= 0:
        raise EmbeddingOutputError("Empty attention mask")

    mask = attention_mask[..., None]

    pooled = (
        (hidden_state.astype(np.float64) * mask)
        .sum(axis=1)[0] / token_count
    )

    norm = float(np.linalg.norm(pooled))

    if not np.isfinite(norm) or norm <= 0:
        raise EmbeddingOutputError("Invalid embedding norm")

    embedding = pooled / norm

    if not np.isfinite(embedding).all():
        raise EmbeddingOutputError("Non-finite embedding")

    return embedding.tolist()
