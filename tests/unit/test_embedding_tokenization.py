from unittest.mock import Mock

import numpy as np
import pytest

from citymind.embeddings.errors import EmbeddingInputError
from citymind.embeddings.tokenization import prepare_model_inputs


def make_tokenizer(count=5):
    tokenizer = Mock()
    tokenizer.truncation = None
    tokenizer.padding = None
    tokenizer.encode.return_value.ids = [1] * count
    tokenizer.encode.return_value.attention_mask = [1] * count
    tokenizer.encode.return_value.type_ids = [0] * count
    return tokenizer


@pytest.mark.parametrize("role,prefix", [
    (True, "query: "),
    (False, "passage: "),
])
def test_prefix(role, prefix):
    tokenizer = make_tokenizer()
    prepare_model_inputs(tokenizer, "İzmir", is_query=role)
    tokenizer.encode.assert_called_once_with(
        prefix + "İzmir", add_special_tokens=True
    )


def test_tensor_format():
    result = prepare_model_inputs(
        make_tokenizer(), "test", is_query=True
    )
    assert set(result) == {
        "input_ids", "attention_mask", "token_type_ids"
    }
    for tensor in result.values():
        assert tensor.shape == (1, 5)
        assert tensor.dtype == np.int64


@pytest.mark.parametrize("count", [1, 512])
def test_valid_token_count(count):
    result = prepare_model_inputs(
        make_tokenizer(count), "test", is_query=True
    )
    assert result["input_ids"].shape == (1, count)


@pytest.mark.parametrize("count", [0, 513])
def test_invalid_token_count(count):
    with pytest.raises(EmbeddingInputError):
        prepare_model_inputs(
            make_tokenizer(count), "test", is_query=True
        )


@pytest.mark.parametrize("setting", ["truncation", "padding"])
def test_unsafe_tokenizer_setting(setting):
    tokenizer = make_tokenizer()
    setattr(tokenizer, setting, {})
    with pytest.raises(EmbeddingInputError):
        prepare_model_inputs(tokenizer, "test", is_query=True)
    tokenizer.encode.assert_not_called()


def test_tokenization_failure():
    tokenizer = make_tokenizer()
    tokenizer.encode.side_effect = RuntimeError("Failure")
    with pytest.raises(EmbeddingInputError):
        prepare_model_inputs(tokenizer, "test", is_query=True)


def test_invalid_role():
    tokenizer = make_tokenizer()
    with pytest.raises(EmbeddingInputError):
        prepare_model_inputs(tokenizer, "test", is_query=1)
    tokenizer.encode.assert_not_called()


def test_invalid_text():
    tokenizer = make_tokenizer()
    with pytest.raises(EmbeddingInputError):
        prepare_model_inputs(tokenizer, "", is_query=True)
    tokenizer.encode.assert_not_called()
