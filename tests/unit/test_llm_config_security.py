"""
CITYMIND centralized LLM configuration security tests.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from citymind.llm.config import (
    LLMConfig,
    validate_llm_config,
)
from citymind.llm.errors import LLMConfigurationError
from citymind.llm.gemini_provider import GeminiProvider


@pytest.fixture
def config():
    return LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-secret-key",
    )


INVALID_CONFIGURATIONS = [
    ("enabled", "false"),
    ("enabled", 1),
    ("enabled", None),
    ("provider", "unknown"),
    ("provider", None),
    ("provider", []),
    ("model", ""),
    ("model", "   "),
    ("model", None),
    ("model", 123),
    ("api_key", ""),
    ("api_key", "   "),
    ("api_key", None),
    ("api_key", 123),
    ("timeout_seconds", True),
    ("timeout_seconds", -1),
    ("timeout_seconds", 0),
    ("timeout_seconds", 121),
    ("timeout_seconds", float("nan")),
    ("timeout_seconds", float("inf")),
    ("timeout_seconds", float("-inf")),
    ("timeout_seconds", 10**400),
    ("timeout_seconds", "30"),
    ("timeout_seconds", None),
    ("max_output_tokens", True),
    ("max_output_tokens", 0),
    ("max_output_tokens", -1),
    ("max_output_tokens", 4097),
    ("max_output_tokens", 1.5),
    ("max_output_tokens", "1024"),
    ("max_output_tokens", None),
]


@pytest.mark.parametrize(
    "field,value",
    INVALID_CONFIGURATIONS,
)
def test_invalid_configuration_rejected(
    config, field, value
):
    invalid_config = replace(
        config, **{field: value}
    )

    with pytest.raises(LLMConfigurationError):
        validate_llm_config(invalid_config)


@pytest.mark.parametrize(
    "field,value",
    INVALID_CONFIGURATIONS,
)
def test_invalid_config_prevents_client_creation(
    config, field, value
):
    invalid_config = replace(
        config, **{field: value}
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:

        with pytest.raises(LLMConfigurationError):
            GeminiProvider(invalid_config)

        client.assert_not_called()


@pytest.mark.parametrize(
    "invalid_config",
    [None, {}, [], "invalid", 123],
)
def test_invalid_configuration_object(
    invalid_config,
):
    with pytest.raises(LLMConfigurationError):
        validate_llm_config(invalid_config)


@pytest.mark.parametrize(
    "timeout",
    [0.001, 1, 30.0, 120],
)
def test_valid_timeout_boundaries(config, timeout):
    valid_config = replace(
        config, timeout_seconds=timeout
    )

    assert validate_llm_config(valid_config) is None


@pytest.mark.parametrize(
    "tokens",
    [1, 1024, 4096],
)
def test_valid_token_boundaries(config, tokens):
    valid_config = replace(
        config, max_output_tokens=tokens
    )

    assert validate_llm_config(valid_config) is None


@pytest.mark.parametrize(
    "provider",
    ["gemini", "groq"],
)
def test_supported_provider_configuration(
    config, provider
):
    valid_config = replace(
        config, provider=provider
    )

    assert validate_llm_config(valid_config) is None


def test_disabled_configuration_without_credentials():
    config = LLMConfig(
        enabled=False,
        provider="gemini",
        model="",
        api_key="",
    )

    assert validate_llm_config(config) is None

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:

        with pytest.raises(LLMConfigurationError):
            GeminiProvider(config)

        client.assert_not_called()


def test_groq_config_cannot_initialize_gemini(config):
    groq_config = replace(
        config, provider="groq"
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:

        with pytest.raises(LLMConfigurationError):
            GeminiProvider(groq_config)

        client.assert_not_called()


def test_validation_error_does_not_expose_api_key(
    config,
):
    invalid_config = replace(
        config, timeout_seconds=-1
    )

    with pytest.raises(LLMConfigurationError) as exc:
        validate_llm_config(invalid_config)

    assert config.api_key not in str(exc.value)
    assert config.api_key not in repr(exc.value)


def test_configuration_repr_hides_api_key(config):
    assert config.api_key not in repr(config)
