"""
CITYMIND Gemini provider unit tests.

All provider requests are mocked.
No real API calls are permitted.
"""

from dataclasses import replace
from unittest.mock import patch, MagicMock

import pytest

from google.genai import errors

from citymind.llm.config import LLMConfig
from citymind.llm.provider import LLMProvider
from citymind.llm.gemini_provider import GeminiProvider
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)


@pytest.fixture
def config():
    return LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-secret-key",
        timeout_seconds=15.0,
        max_output_tokens=512,
    )


@pytest.fixture(autouse=True)
def mock_client():
    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:
        yield client


@pytest.fixture
def provider(config):
    return GeminiProvider(config)


@pytest.fixture
def generate_mock(mock_client):
    return (
        mock_client.return_value.models.generate_content
    )


def test_provider_protocol(provider):
    assert isinstance(provider, LLMProvider)


def test_client_configuration(provider, mock_client):
    mock_client.assert_called_once()

    kwargs = mock_client.call_args.kwargs

    assert kwargs["api_key"] == "mock-secret-key"
    assert kwargs["vertexai"] is False


def test_client_timeout(provider, mock_client):
    options = mock_client.call_args.kwargs["http_options"]

    assert options.timeout == 15000


def test_client_retry_limit(provider, mock_client):
    options = mock_client.call_args.kwargs["http_options"]

    assert options.retry_options.attempts == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": False},
        {"provider": "groq"},
        {"api_key": ""},
        {"model": ""},
    ],
)
def test_invalid_configuration(
    config, mock_client, changes
):
    invalid_config = replace(config, **changes)

    with pytest.raises(LLMConfigurationError):
        GeminiProvider(invalid_config)

    mock_client.assert_not_called()


def test_successful_generation(provider, generate_mock):
    generate_mock.return_value = MagicMock(
        text="Environmental analysis"
    )

    assert provider.generate("Analyze PM2.5") == (
        "Environmental analysis"
    )

    generate_mock.assert_called_once()


def test_response_whitespace_removed(
    provider, generate_mock
):
    generate_mock.return_value = MagicMock(
        text="  Environmental analysis  \n"
    )

    assert provider.generate("Analyze data") == (
        "Environmental analysis"
    )


def test_generation_parameters(provider, generate_mock):
    generate_mock.return_value = MagicMock(text="Analysis")

    provider.generate(
        "Analyze PM2.5",
        system_prompt="You are CITYMIND.",
    )

    kwargs = generate_mock.call_args.kwargs

    assert kwargs["model"] == "mock-model"
    assert kwargs["contents"] == "Analyze PM2.5"

    generation_config = kwargs["config"]

    assert generation_config.max_output_tokens == 512
    assert generation_config.temperature == 0.0
    assert generation_config.system_instruction == (
        "You are CITYMIND."
    )


def test_automatic_function_calling_disabled(
    provider, generate_mock
):
    generate_mock.return_value = MagicMock(text="Analysis")

    provider.generate("Analyze data")

    generation_config = (
        generate_mock.call_args.kwargs["config"]
    )

    assert (
        generation_config.automatic_function_calling.disable
        is True
    )


@pytest.mark.parametrize(
    "prompt",
    ["", "   ", None, 123, {}, []],
)
def test_invalid_prompt(
    provider, generate_mock, prompt
):
    with pytest.raises(LLMConfigurationError):
        provider.generate(prompt)

    generate_mock.assert_not_called()


@pytest.mark.parametrize(
    "system_prompt",
    ["", "   ", 123, {}, []],
)
def test_invalid_system_prompt(
    provider, generate_mock, system_prompt
):
    with pytest.raises(LLMConfigurationError):
        provider.generate(
            "Analyze data",
            system_prompt=system_prompt,
        )

    generate_mock.assert_not_called()


@pytest.mark.parametrize(
    "response_text",
    ["", "   ", None, 123, {}, []],
)
def test_invalid_response(
    provider, generate_mock, response_text
):
    generate_mock.return_value = MagicMock(
        text=response_text
    )

    with pytest.raises(LLMResponseError):
        provider.generate("Analyze data")


def test_unreadable_response(provider, generate_mock):
    class InvalidResponse:
        @property
        def text(self):
            raise ValueError("Invalid response structure")

    generate_mock.return_value = InvalidResponse()

    with pytest.raises(LLMResponseError):
        provider.generate("Analyze data")


def test_timeout_error(provider, generate_mock):
    generate_mock.side_effect = TimeoutError(
        "Sensitive timeout details"
    )

    with pytest.raises(LLMTimeoutError):
        provider.generate("Analyze data")

    generate_mock.assert_called_once()


def test_connection_error(provider, generate_mock):
    generate_mock.side_effect = ConnectionError(
        "Sensitive connection details"
    )

    with pytest.raises(LLMConnectionError):
        provider.generate("Analyze data")


@pytest.mark.parametrize("status_code", [408, 504])
def test_api_timeout_error(
    provider, generate_mock, status_code
):
    generate_mock.side_effect = errors.APIError(
        status_code,
        {
            "error": {
                "code": status_code,
                "message": "Request timeout",
            }
        },
    )

    with pytest.raises(LLMTimeoutError):
        provider.generate("Analyze data")


@pytest.mark.parametrize(
    "status_code", [400, 401, 403, 429, 500, 503]
)
def test_api_request_error(
    provider, generate_mock, status_code
):
    generate_mock.side_effect = errors.APIError(
        status_code,
        {
            "error": {
                "code": status_code,
                "message": "Sensitive provider details",
            }
        },
    )

    with pytest.raises(LLMConnectionError):
        provider.generate("Analyze data")

    generate_mock.assert_called_once()


def test_sensitive_error_details_hidden(
    provider, generate_mock
):
    generate_mock.side_effect = RuntimeError(
        "API key: mock-secret-key"
    )

    with pytest.raises(LLMConnectionError) as exc:
        provider.generate("Analyze data")

    assert "mock-secret-key" not in str(exc.value)
    assert exc.value.__suppress_context__ is True
