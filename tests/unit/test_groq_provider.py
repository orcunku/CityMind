"""Unit tests for the Groq LLM provider."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from citymind.llm.config import LLMConfig
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
)
from citymind.llm.groq_provider import GroqProvider


@pytest.fixture
def config():
    return LLMConfig(
        enabled=True,
        provider="groq",
        model="openai/gpt-oss-120b",
        api_key="citymind-test-fake-key",
    )


@pytest.fixture
def mock_client():
    with patch(
        "citymind.llm.groq_provider.groq.Groq"
    ) as client:
        yield client


def test_client_initialization(config, mock_client):
    with GroqProvider(config) as provider:
        assert provider.closed is False

        mock_client.assert_called_once_with(
            api_key=config.api_key,
            base_url="https://api.groq.com",
            timeout=config.timeout_seconds,
            max_retries=0,
        )


def test_disabled_configuration(config, mock_client):
    with pytest.raises(LLMConfigurationError):
        GroqProvider(replace(config, enabled=False))

    mock_client.assert_not_called()


def test_wrong_provider(config, mock_client):
    with pytest.raises(LLMConfigurationError):
        GroqProvider(replace(config, provider="gemini"))

    mock_client.assert_not_called()


def test_missing_api_key(config, mock_client):
    with pytest.raises(LLMConfigurationError):
        GroqProvider(replace(config, api_key=""))

    mock_client.assert_not_called()


def test_missing_model(config, mock_client):
    with pytest.raises(LLMConfigurationError):
        GroqProvider(replace(config, model=""))

    mock_client.assert_not_called()


def test_initialization_failure(config, mock_client):
    mock_client.side_effect = RuntimeError(config.api_key)

    with pytest.raises(LLMConnectionError) as exc:
        GroqProvider(config)

    assert str(exc.value) == "Unable to initialize Groq client"
    assert config.api_key not in str(exc.value)


def test_context_manager_closes_client(config, mock_client):
    with GroqProvider(config) as provider:
        assert provider.closed is False

    assert provider.closed is True
    mock_client.return_value.close.assert_called_once()


def test_close_is_idempotent(config, mock_client):
    provider = GroqProvider(config)

    provider.close()
    provider.close()

    assert provider.closed is True
    mock_client.return_value.close.assert_called_once()


@pytest.fixture
def provider(config, mock_client):
    with GroqProvider(config) as instance:
        yield instance


@pytest.fixture
def generate_mock(mock_client):
    from types import SimpleNamespace

    generate = mock_client.return_value.chat.completions.create

    generate.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="  CITYMIND_OK  "
                ),
            )
        ]
    )

    return generate


def test_generate_success(provider, generate_mock):
    assert provider.generate("Analyze data") == "CITYMIND_OK"


def test_generate_request_parameters(
    provider, generate_mock, config
):
    provider.generate("Analyze data")

    generate_mock.assert_called_once_with(
        model=config.model,
        messages=[
            {"role": "user", "content": "Analyze data"}
        ],
        temperature=0.0,
        max_completion_tokens=config.max_output_tokens,
        n=1,
        stream=False,
    )


def test_generate_with_system_prompt(
    provider, generate_mock
):
    provider.generate(
        "Analyze data",
        system_prompt="You are CITYMIND",
    )

    messages = generate_mock.call_args.kwargs["messages"]

    assert messages == [
        {"role": "system", "content": "You are CITYMIND"},
        {"role": "user", "content": "Analyze data"},
    ]


@pytest.mark.parametrize(
    "prompt", ["", "   ", None, 123, [], {}]
)
def test_invalid_prompt(provider, generate_mock, prompt):
    with pytest.raises(LLMConfigurationError):
        provider.generate(prompt)

    generate_mock.assert_not_called()


@pytest.mark.parametrize(
    "system_prompt", ["", "   ", 123, [], {}]
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


def test_generate_after_close(provider, generate_mock):
    provider.close()

    with pytest.raises(LLMConfigurationError):
        provider.generate("Analyze data")

    generate_mock.assert_not_called()


@pytest.fixture
def provider(config, mock_client):
    with GroqProvider(config) as instance:
        yield instance


@pytest.fixture
def generate_mock(mock_client):
    from types import SimpleNamespace

    generate = mock_client.return_value.chat.completions.create

    generate.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="  CITYMIND_OK  "
                ),
            )
        ]
    )

    return generate


def test_generate_success(provider, generate_mock):
    assert provider.generate("Analyze data") == "CITYMIND_OK"


def test_generate_request_parameters(
    provider, generate_mock, config
):
    provider.generate("Analyze data")

    generate_mock.assert_called_once_with(
        model=config.model,
        messages=[
            {"role": "user", "content": "Analyze data"}
        ],
        temperature=0.0,
        max_completion_tokens=config.max_output_tokens,
        n=1,
        stream=False,
    )


def test_generate_with_system_prompt(
    provider, generate_mock
):
    provider.generate(
        "Analyze data",
        system_prompt="You are CITYMIND",
    )

    messages = generate_mock.call_args.kwargs["messages"]

    assert messages == [
        {"role": "system", "content": "You are CITYMIND"},
        {"role": "user", "content": "Analyze data"},
    ]


@pytest.mark.parametrize(
    "prompt", ["", "   ", None, 123, [], {}]
)
def test_invalid_prompt(provider, generate_mock, prompt):
    with pytest.raises(LLMConfigurationError):
        provider.generate(prompt)

    generate_mock.assert_not_called()


@pytest.mark.parametrize(
    "system_prompt", ["", "   ", 123, [], {}]
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


def test_generate_after_close(provider, generate_mock):
    provider.close()

    with pytest.raises(LLMConfigurationError):
        provider.generate("Analyze data")

    generate_mock.assert_not_called()


@pytest.mark.parametrize(
    "finish_reason",
    ["length", "content_filter", "tool_calls",
     "function_call", None, "unknown"],
)
def test_invalid_finish_reason(
    provider, generate_mock, finish_reason
):
    from citymind.llm.errors import LLMResponseError

    generate_mock.return_value.choices[0].finish_reason = (
        finish_reason
    )

    with pytest.raises(
        LLMResponseError,
        match="Groq response did not complete normally",
    ):
        provider.generate("Analyze data")


@pytest.mark.parametrize(
    "content", ["", "   ", None, 123, [], {}]
)
def test_invalid_response_content(
    provider, generate_mock, content
):
    from citymind.llm.errors import LLMResponseError

    generate_mock.return_value.choices[0].message.content = (
        content
    )

    with pytest.raises(
        LLMResponseError,
        match="Groq returned an empty response",
    ):
        provider.generate("Analyze data")


def test_empty_choices(provider, generate_mock):
    from citymind.llm.errors import LLMResponseError

    generate_mock.return_value.choices = []

    with pytest.raises(
        LLMResponseError,
        match="Groq returned an invalid choice count",
    ):
        provider.generate("Analyze data")


def test_multiple_choices(provider, generate_mock):
    from citymind.llm.errors import LLMResponseError

    generate_mock.return_value.choices *= 2

    with pytest.raises(
        LLMResponseError,
        match="Groq returned an invalid choice count",
    ):
        provider.generate("Analyze data")


def test_unreadable_response(provider, generate_mock):
    from types import SimpleNamespace
    from citymind.llm.errors import LLMResponseError

    class InvalidMessage:
        @property
        def content(self):
            raise ValueError("Invalid response structure")

    generate_mock.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=InvalidMessage(),
            )
        ]
    )

    with pytest.raises(
        LLMResponseError,
        match="Unable to read Groq response",
    ):
        provider.generate("Analyze data")


def test_api_timeout(provider, generate_mock):
    import groq
    import httpx
    from citymind.llm.errors import LLMTimeoutError

    generate_mock.side_effect = groq.APITimeoutError(
        request=httpx.Request("POST", "https://example.com")
    )

    with pytest.raises(
        LLMTimeoutError,
        match="^Groq request timed out$",
    ):
        provider.generate("Analyze data")

    generate_mock.assert_called_once()


def test_api_connection_error(provider, generate_mock):
    import groq
    import httpx

    generate_mock.side_effect = groq.APIConnectionError(
        request=httpx.Request("POST", "https://example.com")
    )

    with pytest.raises(LLMConnectionError):
        provider.generate("Analyze data")

    generate_mock.assert_called_once()


@pytest.mark.parametrize(
    "status_code", [400, 401, 403, 404, 500, 502, 503]
)
def test_api_status_error(
    provider, generate_mock, config, status_code
):
    import groq
    import httpx

    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://example.com"),
    )

    generate_mock.side_effect = groq.APIStatusError(
        config.api_key,
        response=response,
        body=None,
    )

    with pytest.raises(LLMConnectionError) as exc:
        provider.generate("Analyze data")

    assert str(exc.value) == "Groq API request failed"
    assert config.api_key not in str(exc.value)
    generate_mock.assert_called_once()


@pytest.mark.parametrize("status_code", [408, 504])
def test_http_timeout(provider, generate_mock, status_code):
    import groq
    import httpx
    from citymind.llm.errors import LLMTimeoutError

    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://example.com"),
    )

    generate_mock.side_effect = groq.APIStatusError(
        "Timeout", response=response, body=None
    )

    with pytest.raises(LLMTimeoutError):
        provider.generate("Analyze data")


def test_rate_limit_error(provider, generate_mock):
    import groq
    import httpx
    from citymind.llm.errors import LLMRateLimitError

    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.com"),
    )

    generate_mock.side_effect = groq.RateLimitError(
        "Rate limit exceeded",
        response=response,
        body=None,
    )

    with pytest.raises(
        LLMRateLimitError,
        match="^Groq rate limit or quota exceeded$",
    ):
        provider.generate("Analyze data")

    generate_mock.assert_called_once()


def test_unexpected_api_error(
    provider, generate_mock, config
):
    generate_mock.side_effect = RuntimeError(config.api_key)

    with pytest.raises(LLMConnectionError) as exc:
        provider.generate("Analyze data")

    assert str(exc.value) == "Unable to complete Groq request"
    assert config.api_key not in str(exc.value)


def test_close_failure(config, mock_client):
    provider = GroqProvider(config)
    mock_client.return_value.close.side_effect = (
        RuntimeError(config.api_key)
    )

    with pytest.raises(LLMConnectionError) as exc:
        provider.close()

    assert str(exc.value) == "Unable to close Groq client"
    assert config.api_key not in str(exc.value)
    assert provider.closed is True


def test_enter_after_close(config, mock_client):
    provider = GroqProvider(config)
    provider.close()

    with pytest.raises(
        LLMConfigurationError,
        match="Groq provider is closed",
    ):
        with provider:
            pytest.fail("Closed provider was entered")


def test_context_preserves_original_error(
    config, mock_client
):
    mock_client.return_value.close.side_effect = (
        RuntimeError("Close failed")
    )

    with pytest.raises(ValueError, match="Original error"):
        with GroqProvider(config):
            raise ValueError("Original error")

    mock_client.return_value.close.assert_called_once()


def test_context_propagates_close_failure(
    config, mock_client
):
    mock_client.return_value.close.side_effect = (
        RuntimeError("Close failed")
    )

    with pytest.raises(
        LLMConnectionError,
        match="Unable to close Groq client",
    ):
        with GroqProvider(config):
            pass


def test_provider_protocol(provider):
    from citymind.llm.provider import LLMProvider

    assert isinstance(provider, LLMProvider)


def test_api_key_hidden_in_config_repr(config):
    assert config.api_key not in repr(config)


def test_exception_context_suppressed(
    provider, generate_mock, config
):
    generate_mock.side_effect = RuntimeError(config.api_key)

    with pytest.raises(LLMConnectionError) as exc:
        provider.generate("Analyze data")

    assert exc.value.__suppress_context__ is True
    assert config.api_key not in str(exc.value)
