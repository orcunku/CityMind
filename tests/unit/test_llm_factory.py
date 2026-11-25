"""Unit tests for the central LLM provider factory."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from citymind.llm.config import LLMConfig
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
)
from citymind.llm.factory import create_llm_provider


@pytest.fixture
def config():
    return LLMConfig(
        enabled=True,
        provider="groq",
        model="openai/gpt-oss-120b",
        api_key="citymind-test-fake-key",
    )


@pytest.mark.parametrize(
    "provider,model,class_name",
    [
        ("groq", "openai/gpt-oss-120b", "GroqProvider"),
        ("gemini", "gemini-3.1-flash-lite", "GeminiProvider"),
    ],
)
def test_select_correct_provider(
    config, provider, model, class_name
):
    selected = replace(
        config, provider=provider, model=model
    )

    path = f"citymind.llm.{provider}_provider.{class_name}"

    with patch(path) as constructor:
        result = create_llm_provider(selected)

        constructor.assert_called_once_with(selected)
        assert result is constructor.return_value


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": False},
        {"provider": "unknown"},
        {"api_key": ""},
        {"model": ""},
        {"timeout_seconds": 0},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_configuration(config, changes):
    with patch(
        "citymind.llm.groq_provider.GroqProvider"
    ) as groq:
        with patch(
            "citymind.llm.gemini_provider.GeminiProvider"
        ) as gemini:
            with pytest.raises(LLMConfigurationError):
                create_llm_provider(
                    replace(config, **changes)
                )

            groq.assert_not_called()
            gemini.assert_not_called()


def test_invalid_config_object():
    with pytest.raises(LLMConfigurationError):
        create_llm_provider(None)


def test_no_automatic_fallback(config):
    with patch(
        "citymind.llm.groq_provider.GroqProvider"
    ) as groq:
        with patch(
            "citymind.llm.gemini_provider.GeminiProvider"
        ) as gemini:
            groq.side_effect = LLMConnectionError(
                "Provider unavailable"
            )

            with pytest.raises(LLMConnectionError):
                create_llm_provider(config)

            groq.assert_called_once_with(config)
            gemini.assert_not_called()
