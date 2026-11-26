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


@pytest.fixture(params=["groq", "gemini"])
def lifecycle_provider(config, request):
    provider = request.param

    if provider == "groq":
        model = "openai/gpt-oss-120b"
        path = "citymind.llm.groq_provider.groq.Groq"
    else:
        model = "gemini-3.1-flash-lite"
        path = "citymind.llm.gemini_provider.genai.Client"

    selected = replace(
        config, provider=provider, model=model
    )

    with patch(path) as constructor:
        instance = create_llm_provider(selected)
        yield instance, constructor.return_value


def test_factory_provider_implements_lifecycle(
    lifecycle_provider,
):
    from citymind.llm.provider import LLMProvider

    provider, client = lifecycle_provider

    assert isinstance(provider, LLMProvider)
    assert provider.closed is False
    client.close.assert_not_called()


def test_factory_provider_context_closes_client(
    lifecycle_provider,
):
    provider, client = lifecycle_provider

    with provider as active:
        assert active is provider
        assert active.closed is False
        client.close.assert_not_called()

    assert provider.closed is True
    client.close.assert_called_once_with()


def test_factory_provider_close_is_idempotent(
    lifecycle_provider,
):
    provider, client = lifecycle_provider

    provider.close()
    provider.close()

    assert provider.closed is True
    client.close.assert_called_once_with()

def test_factory_provider_closes_on_error(
    lifecycle_provider,
):
    provider, client = lifecycle_provider

    with pytest.raises(RuntimeError, match="Test failure"):
        with provider:
            raise RuntimeError("Test failure")

    assert provider.closed is True
    client.close.assert_called_once_with()

def test_factory_provider_rejects_closed_context(
    lifecycle_provider,
):
    provider, client = lifecycle_provider
    provider.close()

    with pytest.raises(LLMConfigurationError):
        with provider:
            pytest.fail("Closed provider was reopened")

    client.close.assert_called_once_with()

def test_factory_provider_rejects_closed_generation(
    lifecycle_provider,
):
    provider, client = lifecycle_provider
    provider.close()

    with pytest.raises(LLMConfigurationError):
        provider.generate("Analyze PM2.5")

    client.chat.completions.create.assert_not_called()
    client.models.generate_content.assert_not_called()
    client.close.assert_called_once_with()


def test_factory_provider_cleanup_failure(
    lifecycle_provider,
):
    provider, client = lifecycle_provider
    client.close.side_effect = RuntimeError("SDK failure")

    with pytest.raises(LLMConnectionError):
        with provider:
            pass

    assert provider.closed is True

    with pytest.raises(LLMConfigurationError):
        provider.generate("Analyze PM2.5")

    client.close.assert_called_once_with()


def test_factory_provider_preserves_original_error(
    lifecycle_provider,
):
    provider, client = lifecycle_provider
    client.close.side_effect = RuntimeError("SDK failure")

    original = ValueError("Original operation failure")

    with pytest.raises(ValueError) as captured:
        with provider:
            raise original

    assert captured.value is original
    assert provider.closed is True
    client.close.assert_called_once_with()


def test_factory_provider_cleanup_failure(
    lifecycle_provider,
):
    provider, client = lifecycle_provider
    client.close.side_effect = RuntimeError("SDK failure")

    with pytest.raises(LLMConnectionError):
        with provider:
            pass

    assert provider.closed is True

    with pytest.raises(LLMConfigurationError):
        provider.generate("Analyze PM2.5")

    client.close.assert_called_once_with()


def test_factory_provider_preserves_original_error(
    lifecycle_provider,
):
    provider, client = lifecycle_provider
    client.close.side_effect = RuntimeError("SDK failure")

    original = ValueError("Original operation failure")

    with pytest.raises(ValueError) as captured:
        with provider:
            raise original

    assert captured.value is original
    assert provider.closed is True
    client.close.assert_called_once_with()
