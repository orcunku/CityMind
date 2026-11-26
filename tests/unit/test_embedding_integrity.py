import hashlib

import pytest

from citymind.embeddings.errors import EmbeddingModelError
from citymind.embeddings.integrity import verify_model_file


DATA = b"CITYMIND test model"
DIGEST = hashlib.sha256(DATA).hexdigest()


def test_valid_model_file(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(DATA)
    assert verify_model_file(
        path, len(DATA), DIGEST
    ) == path.resolve()


def test_missing_model_file(tmp_path):
    with pytest.raises(EmbeddingModelError):
        verify_model_file(
            tmp_path / "missing.onnx", len(DATA), DIGEST
        )


def test_directory_rejected(tmp_path):
    with pytest.raises(EmbeddingModelError):
        verify_model_file(tmp_path, len(DATA), DIGEST)


def test_wrong_file_size(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(DATA)
    with pytest.raises(EmbeddingModelError, match="size mismatch"):
        verify_model_file(path, len(DATA) + 1, DIGEST)


def test_wrong_sha256(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(DATA)
    with pytest.raises(EmbeddingModelError, match="SHA256 mismatch"):
        verify_model_file(path, len(DATA), "0" * 64)


def test_unreadable_model_file(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "model.onnx"
    path.write_bytes(DATA)

    def deny_access(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr(Path, "open", deny_access)

    with pytest.raises(EmbeddingModelError):
        verify_model_file(path, len(DATA), DIGEST)


def test_valid_model_symlink(tmp_path):
    target = tmp_path / "model.onnx"
    target.write_bytes(DATA)

    link = tmp_path / "linked.onnx"
    link.symlink_to(target)

    assert verify_model_file(
        link, len(DATA), DIGEST
    ) == target.resolve()


def test_broken_model_symlink(tmp_path):
    link = tmp_path / "linked.onnx"
    link.symlink_to(tmp_path / "missing.onnx")

    with pytest.raises(EmbeddingModelError):
        verify_model_file(link, len(DATA), DIGEST)
