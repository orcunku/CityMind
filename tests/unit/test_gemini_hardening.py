"""
CITYMIND Gemini transport and rate limit tests.
"""

from unittest.mock import patch

import httpx
import httpx2
import pytest

from google.genai import errors

from citymind.llm.config import LLMConfig
from citymind.llm.gemini_provider import GeminiProvider
from citymind.llm.errors import (
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)


@pytest.fixture
def provider_and_generate():
    config = LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-secret-key",
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client_class:

        provider = GeminiProvider(config)

        generate = (
            client_class.return_value
            .models.generate_content
        )

        yield provider, generate


@pytest.mark.parametrize(
    "timeout_class",
    [
        TimeoutError,
        httpx.TimeoutException,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx2.TimeoutException,
        httpx2.ConnectTimeout,
        httpx2.ReadTimeout,
        httpx2.WriteTimeout,
        httpx2.PoolTimeout,
    ],
)
def test_transport_timeout(
    provider_and_generate, timeout_class
):
    provider, generate = provider_and_generate

    generate.side_effect = timeout_class(
        "Sensitive transport details"
    )

    with pytest.raises(LLMTimeoutError) as exc:
        provider.generate("Analyze PM2.5")

    assert str(exc.value) == (
        "Gemini request timed out"
    )

    assert exc.value.__suppress_context__ is True

    generate.assert_called_once()


def test_rate_limit_error_hierarchy():
    assert issubclass(
        LLMRateLimitError, LLMConnectionError
    )

    assert issubclass(
        LLMRateLimitError, LLMError
    )


def test_rate_limit_is_classified_separately(
    provider_and_generate,
):
    provider, generate = provider_and_generate

    generate.side_effect = errors.APIError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Sensitive quota details",
            }
        },
    )

    with pytest.raises(LLMRateLimitError) as exc:
        provider.generate("Analyze PM2.5")

    assert str(exc.value) == (
        "Gemini rate limit or quota exceeded"
    )

    assert exc.value.__suppress_context__ is True

    generate.assert_called_once()


@pytest.mark.parametrize("status_code", [408, 504])
def test_api_timeout_classification(
    provider_and_generate, status_code
):
    provider, generate = provider_and_generate

    generate.side_effect = errors.APIError(
        status_code,
        {
            "error": {
                "code": status_code,
                "message": "Sensitive timeout details",
            }
        },
    )

    with pytest.raises(LLMTimeoutError) as exc:
        provider.generate("Analyze PM2.5")

    assert str(exc.value) == (
        "Gemini request timed out"
    )

    generate.assert_called_once()


@pytest.mark.parametrize(
    "status_code", [400, 401, 403, 500, 503]
)
def test_other_api_errors_are_connection_errors(
    provider_and_generate, status_code
):
    provider, generate = provider_and_generate

    generate.side_effect = errors.APIError(
        status_code,
        {
            "error": {
                "code": status_code,
                "message": "Sensitive API details",
            }
        },
    )

    with pytest.raises(LLMConnectionError) as exc:
        provider.generate("Analyze PM2.5")

    assert type(exc.value) is LLMConnectionError

    assert str(exc.value) == (
        "Gemini API request failed"
    )

    generate.assert_called_once()
