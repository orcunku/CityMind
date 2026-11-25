"""
CITYMIND LLM configuration unit tests.
"""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from citymind.llm.config import load_llm_config
from citymind.llm.errors import LLMConfigurationError


@pytest.fixture(autouse=True)
def isolated_environment():
    with patch.dict("os.environ", {}, clear=True):
        yield


def test_llm_disabled_by_default():
    assert load_llm_config().enabled is False


def test_default_provider():
    assert load_llm_config().provider == "gemini"


def test_default_timeout():
    assert load_llm_config().timeout_seconds == 30.0


def test_default_token_limit():
    assert load_llm_config().max_output_tokens == 1024


@pytest.mark.parametrize(
    "provider,key_variable",
    [
        ("gemini", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ],
)
def test_enabled_provider_configuration(
    monkeypatch, provider, key_variable
):
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv(key_variable, "mock-secret")

    config = load_llm_config()

    assert config.enabled is True
    assert config.provider == provider
    assert config.model == "test-model"
    assert config.api_key == "mock-secret"


def test_api_key_hidden_from_repr(monkeypatch):
    monkeypatch.setenv(
        "GEMINI_API_KEY", "mock-secret"
    )

    assert "mock-secret" not in repr(load_llm_config())


def test_configuration_is_immutable():
    config = load_llm_config()

    with pytest.raises(FrozenInstanceError):
        config.enabled = True


@pytest.mark.parametrize(
    "value", ["yes", "1", "invalid", ""]
)
def test_invalid_enabled_value(monkeypatch, value):
    monkeypatch.setenv("LLM_ENABLED", value)

    with pytest.raises(LLMConfigurationError):
        load_llm_config()


@pytest.mark.parametrize(
    "provider", ["unknown", "openai", ""]
)
def test_unsupported_provider(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)

    with pytest.raises(LLMConfigurationError):
        load_llm_config()


def test_enabled_llm_requires_model(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-secret")

    with pytest.raises(
        LLMConfigurationError, match="LLM_MODEL"
    ):
        load_llm_config()


@pytest.mark.parametrize(
    "provider,key_variable",
    [
        ("gemini", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ],
)
def test_enabled_llm_requires_correct_key(
    monkeypatch, provider, key_variable
):
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "test-model")

    with pytest.raises(
        LLMConfigurationError, match=key_variable
    ):
        load_llm_config()


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "121", "nan", "inf", "-inf", "invalid"],
)
def test_invalid_timeout(monkeypatch, value):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", value)

    with pytest.raises(LLMConfigurationError):
        load_llm_config()


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "4097", "1.5", "invalid"],
)
def test_invalid_token_limit(monkeypatch, value):
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", value)

    with pytest.raises(LLMConfigurationError):
        load_llm_config()


@pytest.mark.parametrize("value", ["0.5", "30", "120"])
def test_valid_timeout(monkeypatch, value):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", value)

    assert load_llm_config().timeout_seconds == float(value)


@pytest.mark.parametrize("value", ["1", "1024", "4096"])
def test_valid_token_limit(monkeypatch, value):
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", value)

    assert load_llm_config().max_output_tokens == int(value)


def test_configuration_normalization(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", " TRUE ")
    monkeypatch.setenv("LLM_PROVIDER", " GEMINI ")
    monkeypatch.setenv("LLM_MODEL", " test-model ")
    monkeypatch.setenv("GEMINI_API_KEY", " mock-secret ")

    config = load_llm_config()

    assert config.enabled is True
    assert config.provider == "gemini"
    assert config.model == "test-model"
    assert config.api_key == "mock-secret"


def test_disabled_llm_requires_no_api_key():
    config = load_llm_config()

    assert config.enabled is False
    assert config.api_key == ""
    assert config.model == ""
