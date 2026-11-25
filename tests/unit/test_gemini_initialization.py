"""
CITYMIND Gemini initialization security tests.
"""

from unittest.mock import patch

import pytest

from citymind.llm.config import LLMConfig
from citymind.llm.gemini_provider import GeminiProvider
from citymind.llm.errors import (
    LLMError,
    LLMConnectionError,
    LLMConfigurationError,
)


@pytest.fixture
def config():
    return LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-secret-key",
    )


@pytest.mark.parametrize(
    "error_class",
    [
        RuntimeError,
        ValueError,
        TypeError,
        ConnectionError,
        TimeoutError,
        OSError,
    ],
)
def test_initialization_error_classification(
    config, error_class
):
    with patch(
        "citymind.llm.gemini_provider.genai.Client",
        side_effect=error_class(
            "Sensitive initialization details"
        ),
    ) as client:

        with pytest.raises(LLMConnectionError) as exc:
            GeminiProvider(config)

        assert isinstance(exc.value, LLMError)

        assert str(exc.value) == (
            "Unable to initialize Gemini client"
        )

        client.assert_called_once()


def test_initialization_secret_not_exposed(config):
    with patch(
        "citymind.llm.gemini_provider.genai.Client",
        side_effect=RuntimeError(
            f"Invalid credentials: {config.api_key}"
        ),
    ):
        with pytest.raises(LLMConnectionError) as exc:
            GeminiProvider(config)

        assert config.api_key not in str(exc.value)
        assert config.api_key not in repr(exc.value)

        assert exc.value.__suppress_context__ is True


def test_successful_initialization(config):
    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:

        provider = GeminiProvider(config)

        assert provider.closed is False

        client.assert_called_once()


def test_invalid_config_rejected_before_initialization():
    config = LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-secret-key",
        timeout_seconds=-1,
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:

        with pytest.raises(LLMConfigurationError):
            GeminiProvider(config)

        client.assert_not_called()
