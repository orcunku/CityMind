"""CITYMIND semantic embedding tensor preparation."""

import numpy as np
from tokenizers import Tokenizer

from citymind.embeddings import config
from citymind.embeddings.errors import EmbeddingInputError
from citymind.embeddings.validation import validate_embedding_text


def prepare_model_inputs(
    tokenizer: Tokenizer,
    text: str,
    *,
    is_query: bool,
) -> dict[str, np.ndarray]:
    """Prepare validated ONNX input tensors."""
    text = validate_embedding_text(text)

    if type(is_query) is not bool:
        raise EmbeddingInputError("Invalid embedding input role")

    prefix = (
        config.QUERY_PREFIX
        if is_query
        else config.DOCUMENT_PREFIX
    )

    try:
        if tokenizer.truncation is not None:
            raise EmbeddingInputError("Unexpected tokenizer truncation")

        if tokenizer.padding is not None:
            raise EmbeddingInputError("Unexpected tokenizer padding")

        encoding = tokenizer.encode(
            prefix + text,
            add_special_tokens=True,
        )
    except EmbeddingInputError:
        raise
    except Exception:
        raise EmbeddingInputError("Tokenization failed") from None

    count = len(encoding.ids)

    if not 1 <= count <= config.MAX_INPUT_TOKENS:
        raise EmbeddingInputError("Invalid embedding token count")

    return {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array(
            [encoding.attention_mask], dtype=np.int64
        ),
        "token_type_ids": np.array(
            [encoding.type_ids], dtype=np.int64
        ),
    }
