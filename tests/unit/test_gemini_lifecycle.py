"""
CITYMIND Gemini client lifecycle tests.
"""

from unittest.mock import patch

import pytest

from citymind.llm.config import LLMConfig
from citymind.llm.gemini_provider import GeminiProvider
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
)


@pytest.fixture
def provider_and_client():
    config = LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-key",
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client_class:

        provider = GeminiProvider(config)

        yield provider, client_class.return_value


def test_provider_initially_open(provider_and_client):
    provider, client = provider_and_client

    assert provider.closed is False
    client.close.assert_not_called()


def test_close_releases_client(provider_and_client):
    provider, client = provider_and_client

    provider.close()

    assert provider.closed is True
    client.close.assert_called_once()


def test_close_is_idempotent(provider_and_client):
    provider, client = provider_and_client

    provider.close()
    provider.close()
    provider.close()

    client.close.assert_called_once()


def test_context_manager_returns_provider(
    provider_and_client,
):
    provider, client = provider_and_client

    with provider as active_provider:
        assert active_provider is provider
        assert provider.closed is False


def test_context_manager_closes_client(
    provider_and_client,
):
    provider, client = provider_and_client

    with provider:
        pass

    assert provider.closed is True
    client.close.assert_called_once()


def test_context_manager_closes_after_error(
    provider_and_client,
):
    provider, client = provider_and_client

    with pytest.raises(ValueError, match="Original error"):
        with provider:
            raise ValueError("Original error")

    assert provider.closed is True
    client.close.assert_called_once()


def test_closed_provider_cannot_enter_context(
    provider_and_client,
):
    provider, client = provider_and_client

    provider.close()

    with pytest.raises(LLMConfigurationError):
        with provider:
            pass

    client.close.assert_called_once()


def test_close_failure_is_sanitized(
    provider_and_client,
):
    provider, client = provider_and_client

    client.close.side_effect = RuntimeError(
        "Sensitive client details"
    )

    with pytest.raises(LLMConnectionError) as exc:
        provider.close()

    assert provider.closed is True
    assert "Sensitive client details" not in str(exc.value)


def test_cleanup_failure_preserves_original_error(
    provider_and_client,
):
    provider, client = provider_and_client

    client.close.side_effect = RuntimeError(
        "Cleanup failure"
    )

    with pytest.raises(ValueError, match="Original error"):
        with provider:
            raise ValueError("Original error")

    assert provider.closed is True
    client.close.assert_called_once()
