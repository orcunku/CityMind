"""
CITYMIND closed Gemini client protection tests.
"""

from unittest.mock import patch, MagicMock

import pytest

from google.genai import types

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


def test_closed_provider_rejects_generation(
    provider_and_client,
):
    provider, client = provider_and_client

    provider.close()

    with pytest.raises(
        LLMConfigurationError, match="closed"
    ):
        provider.generate("Analyze PM2.5")

    client.models.generate_content.assert_not_called()


def test_context_exit_prevents_generation(
    provider_and_client,
):
    provider, client = provider_and_client

    with provider:
        pass

    with pytest.raises(
        LLMConfigurationError, match="closed"
    ):
        provider.generate("Analyze data")

    client.models.generate_content.assert_not_called()


def test_failed_close_prevents_generation(
    provider_and_client,
):
    provider, client = provider_and_client

    client.close.side_effect = RuntimeError(
        "Simulated cleanup failure"
    )

    with pytest.raises(LLMConnectionError):
        provider.close()

    with pytest.raises(
        LLMConfigurationError, match="closed"
    ):
        provider.generate("Analyze data")

    client.models.generate_content.assert_not_called()


@pytest.mark.parametrize(
    "prompt", ["", None, "Valid prompt"]
)
def test_closed_state_checked_before_prompt(
    provider_and_client, prompt
):
    provider, client = provider_and_client

    provider.close()

    with pytest.raises(
        LLMConfigurationError, match="closed"
    ):
        provider.generate(prompt)

    client.models.generate_content.assert_not_called()


def test_open_provider_can_generate(
    provider_and_client,
):
    provider, client = provider_and_client

    client.models.generate_content.return_value = (
        MagicMock(
            candidates=[
                MagicMock(
                    finish_reason=types.FinishReason.STOP
                )
            ],
            text="Environmental analysis",
        )
    )

    assert provider.closed is False

    assert provider.generate("Analyze data") == (
        "Environmental analysis"
    )

    client.models.generate_content.assert_called_once()


def test_repeated_generation_after_close_is_blocked(
    provider_and_client,
):
    provider, client = provider_and_client

    provider.close()

    for _ in range(3):
        with pytest.raises(LLMConfigurationError):
            provider.generate("Analyze data")

    client.models.generate_content.assert_not_called()
