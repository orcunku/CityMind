"""CITYMIND secure local ONNX inference."""

import numpy as np
import onnxruntime as ort

from citymind.embeddings import config
from citymind.embeddings.errors import (
    EmbeddingInferenceError,
    EmbeddingOutputError,
)


def run_model_inference(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """Execute local inference and validate model output."""
    try:
        outputs = session.run(
            ["last_hidden_state"],
            inputs,
        )
    except Exception:
        raise EmbeddingInferenceError(
            "Local embedding inference failed"
        ) from None

    if not isinstance(outputs, list) or len(outputs) != 1:
        raise EmbeddingOutputError("Invalid model output count")

    hidden_state = outputs[0]

    expected_shape = (
        1,
        inputs["input_ids"].shape[1],
        config.EMBEDDING_DIMENSION,
    )

    if not isinstance(hidden_state, np.ndarray):
        raise EmbeddingOutputError("Invalid model output type")

    if hidden_state.shape != expected_shape:
        raise EmbeddingOutputError("Invalid model output shape")

    if hidden_state.dtype != np.float32:
        raise EmbeddingOutputError("Invalid model output dtype")

    if not np.isfinite(hidden_state).all():
        raise EmbeddingOutputError("Non-finite model output")

    return hidden_state
