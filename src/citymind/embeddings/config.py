"""CITYMIND local embedding model configuration."""

MODEL_REPOSITORY = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

MODEL_FILENAME = "model_qint8_avx512_vnni.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

MODEL_SHA256 = "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"
TOKENIZER_SHA256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"

MODEL_SIZE_BYTES = 118346824
TOKENIZER_SIZE_BYTES = 17082730

EMBEDDING_DIMENSION = 384
MAX_INPUT_TOKENS = 512
MAX_INPUT_CHARS = 16000

INTRA_OP_THREADS = 2
INTER_OP_THREADS = 1

QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "

from pathlib import Path

from citymind.embeddings.errors import EmbeddingConfigurationError

def resolve_model_directory(
    model_directory: str | Path | None = None,
) -> Path:
    """Resolve an existing local embedding model directory."""
    if model_directory is None:
        directory = (
            Path.home()
            / ".cache/huggingface/hub"
            / "models--intfloat--multilingual-e5-small"
            / "snapshots"
            / MODEL_REVISION
            / "onnx"
        )
    elif isinstance(model_directory, (str, Path)):
        if not str(model_directory).strip():
            raise EmbeddingConfigurationError("Empty model directory")
        directory = Path(model_directory)
    else:
        raise EmbeddingConfigurationError("Invalid model directory")

    try:
        directory = directory.expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise EmbeddingConfigurationError("Invalid model directory")
    except (OSError, RuntimeError, ValueError):
        raise EmbeddingConfigurationError(
            "Local model directory unavailable"
        ) from None

    return directory
