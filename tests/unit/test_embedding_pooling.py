import numpy as np
import pytest

from citymind.embeddings.errors import EmbeddingOutputError
from citymind.embeddings.pooling import pool_and_normalize


def make_hidden():
    return np.ones((1, 3, 384), dtype=np.float32)


def make_mask():
    return np.ones((1, 3), dtype=np.int64)


def test_valid_embedding():
    result = pool_and_normalize(make_hidden(), make_mask())
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_masked_mean_pooling():
    hidden = np.zeros((1, 3, 384), dtype=np.float32)
    hidden[0, 0, 0] = 3
    hidden[0, 1, 1] = 4
    hidden[0, 2, 2] = 1000

    mask = np.array([[1, 1, 0]], dtype=np.int64)
    result = pool_and_normalize(hidden, mask)

    assert result[0] == pytest.approx(0.6)
    assert result[1] == pytest.approx(0.8)
    assert result[2:] == [0.0] * 382


def test_zero_norm():
    hidden = np.zeros((1, 3, 384), dtype=np.float32)
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(hidden, make_mask())


def test_empty_mask():
    mask = np.zeros((1, 3), dtype=np.int64)
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(make_hidden(), mask)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_hidden(value):
    hidden = make_hidden()
    hidden[0, 0, 0] = value
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(hidden, make_mask())


@pytest.mark.parametrize("shape", [
    (1, 3, 383),
    (2, 3, 384),
    (3, 384),
])
def test_invalid_hidden_shape(shape):
    hidden = np.ones(shape, dtype=np.float32)
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(hidden, make_mask())


def test_invalid_hidden_dtype():
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(
            make_hidden().astype(np.float64), make_mask()
        )


def test_invalid_mask_shape():
    mask = np.ones((1, 2), dtype=np.int64)
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(make_hidden(), mask)


def test_invalid_mask_dtype():
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(
            make_hidden(), make_mask().astype(np.float32)
        )


@pytest.mark.parametrize("value", [-1, 2])
def test_invalid_mask_value(value):
    mask = make_mask()
    mask[0, 0] = value
    with pytest.raises(EmbeddingOutputError):
        pool_and_normalize(make_hidden(), mask)


def test_large_finite_values():
    hidden = np.full(
        (1, 3, 384),
        np.finfo(np.float32).max,
        dtype=np.float32,
    )
    result = pool_and_normalize(hidden, make_mask())
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0)
