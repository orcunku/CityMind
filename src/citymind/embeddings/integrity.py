"""CITYMIND local embedding model integrity checks."""

import hashlib
from pathlib import Path

from citymind.embeddings.errors import EmbeddingModelError


def verify_model_file(
    path: Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    """Verify a local model file before loading."""
    try:
        path = path.resolve(strict=True)

        if not path.is_file():
            raise EmbeddingModelError("Invalid model file")

        if path.stat().st_size != expected_size:
            raise EmbeddingModelError("Model file size mismatch")

        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()

        if digest != expected_sha256:
            raise EmbeddingModelError("Model file SHA256 mismatch")

    except (OSError, RuntimeError, ValueError):
        raise EmbeddingModelError(
            "Unable to verify model file"
        ) from None

    return path
