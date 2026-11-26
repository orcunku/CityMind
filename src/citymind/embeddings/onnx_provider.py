"""CITYMIND local ONNX semantic embedding provider."""

from pathlib import Path

import onnxruntime as ort
from tokenizers import Tokenizer

from citymind.embeddings import config
from citymind.embeddings.errors import EmbeddingModelError
from citymind.embeddings.integrity import verify_model_file


class ONNXEmbeddingProvider:
    """Local CPU-based semantic embedding provider."""

    def __init__(
        self,
        model_directory: str | Path | None = None,
    ) -> None:
        directory = config.resolve_model_directory(model_directory)

        model_path = verify_model_file(
            directory / config.MODEL_FILENAME,
            config.MODEL_SIZE_BYTES,
            config.MODEL_SHA256,
        )

        tokenizer_path = verify_model_file(
            directory / config.TOKENIZER_FILENAME,
            config.TOKENIZER_SIZE_BYTES,
            config.TOKENIZER_SHA256,
        )

        try:
            self._tokenizer = Tokenizer.from_file(
                str(tokenizer_path)
            )

            options = ort.SessionOptions()
            options.intra_op_num_threads = config.INTRA_OP_THREADS
            options.inter_op_num_threads = config.INTER_OP_THREADS

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._session.disable_fallback()

        except Exception:
            raise EmbeddingModelError(
                "Unable to initialize local embedding model"
            ) from None

        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise EmbeddingModelError("Unexpected execution provider")

        self._validate_model_interface()

    def _validate_model_interface(self) -> None:
        """Validate the expected ONNX tensor interface."""
        expected_inputs = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }

        inputs = self._session.get_inputs()

        if (
            len(inputs) != 3
            or {item.name for item in inputs} != expected_inputs
        ):
            raise EmbeddingModelError("Unexpected model inputs")

        for item in inputs:
            if item.type != "tensor(int64)" or len(item.shape) != 2:
                raise EmbeddingModelError("Invalid model input tensor")

        outputs = self._session.get_outputs()

        if len(outputs) != 1:
            raise EmbeddingModelError("Unexpected model outputs")

        output = outputs[0]

        if (
            output.name != "last_hidden_state"
            or output.type != "tensor(float)"
            or len(output.shape) != 3
            or output.shape[-1] != config.EMBEDDING_DIMENSION
        ):
            raise EmbeddingModelError("Invalid model output tensor")

    @property
    def model_id(self) -> str:
        """Return the versioned embedding model identity."""
        return (
            f"{config.MODEL_REPOSITORY}"
            f"@{config.MODEL_REVISION}"
            f":{config.MODEL_FILENAME}"
            ":mean-pooling-l2-v1"
        )

    @property
    def dimension(self) -> int:
        """Return the semantic embedding dimension."""
        return config.EMBEDDING_DIMENSION
