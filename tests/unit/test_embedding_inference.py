from unittest.mock import Mock

import numpy as np
import pytest

from citymind.embeddings.errors import (
    EmbeddingInferenceError,
    EmbeddingOutputError,
)
from citymind.embeddings.inference import run_model_inference


def make_inputs():
    return {
        "input_ids": np.ones((1, 5), dtype=np.int64),
        "attention_mask": np.ones((1, 5), dtype=np.int64),
        "token_type_ids": np.zeros((1, 5), dtype=np.int64),
    }


def test_valid_inference():
    session = Mock()
    inputs = make_inputs()
    expected = np.ones((1, 5, 384), dtype=np.float32)
    session.run.return_value = [expected]

    result = run_model_inference(session, inputs)

    assert result is expected
    session.run.assert_called_once_with(
        ["last_hidden_state"], inputs
    )


def test_inference_failure():
    session = Mock()
    session.run.side_effect = RuntimeError("Failure")

    with pytest.raises(EmbeddingInferenceError):
        run_model_inference(session, make_inputs())


@pytest.mark.parametrize("outputs", [None, [], [1, 2], ()])
def test_invalid_output_count(outputs):
    session = Mock()
    session.run.return_value = outputs

    with pytest.raises(EmbeddingOutputError):
        run_model_inference(session, make_inputs())


def test_invalid_output_type():
    session = Mock()
    session.run.return_value = ["invalid"]

    with pytest.raises(EmbeddingOutputError):
        run_model_inference(session, make_inputs())


@pytest.mark.parametrize("shape", [
    (1, 5, 383),
    (2, 5, 384),
    (1, 4, 384),
    (5, 384),
])
def test_invalid_output_shape(shape):
    session = Mock()
    session.run.return_value = [
        np.ones(shape, dtype=np.float32)
    ]

    with pytest.raises(EmbeddingOutputError):
        run_model_inference(session, make_inputs())


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_invalid_output_dtype(dtype):
    session = Mock()
    session.run.return_value = [
        np.ones((1, 5, 384), dtype=dtype)
    ]

    with pytest.raises(EmbeddingOutputError):
        run_model_inference(session, make_inputs())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_output(value):
    session = Mock()
    output = np.ones((1, 5, 384), dtype=np.float32)
    output[0, 0, 0] = value
    session.run.return_value = [output]

    with pytest.raises(EmbeddingOutputError):
        run_model_inference(session, make_inputs())
