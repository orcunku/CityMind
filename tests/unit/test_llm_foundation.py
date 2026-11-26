"""
CITYMIND provider-independent LLM foundation tests.
"""

import pytest

from citymind.llm.provider import LLMProvider
from citymind.llm.errors import (
    LLMError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)


class MockLLMProvider:
    def __init__(self):
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise LLMConfigurationError("Provider is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        return "Mock environmental analysis"


class InvalidProvider:
    pass


def test_provider_implements_protocol():
    assert isinstance(
        MockLLMProvider(),
        LLMProvider,
    )


def test_invalid_provider_is_not_recognized():
    assert not isinstance(
        InvalidProvider(),
        LLMProvider,
    )


def test_provider_generates_text():
    provider = MockLLMProvider()

    response = provider.generate(
        "Analyze PM2.5 measurement."
    )

    assert isinstance(response, str)
    assert response == "Mock environmental analysis"


def test_provider_accepts_system_prompt():
    provider = MockLLMProvider()

    response = provider.generate(
        "Analyze environmental data.",
        system_prompt="You are CITYMIND.",
    )

    assert response == "Mock environmental analysis"


def test_base_error_inherits_exception():
    assert issubclass(LLMError, Exception)


@pytest.mark.parametrize(
    "error_class",
    [
        LLMConfigurationError,
        LLMConnectionError,
        LLMResponseError,
        LLMTimeoutError,
    ],
)
def test_llm_error_hierarchy(error_class):
    assert issubclass(error_class, LLMError)


@pytest.mark.parametrize(
    "error_class",
    [
        LLMConfigurationError,
        LLMConnectionError,
        LLMResponseError,
        LLMTimeoutError,
    ],
)
def test_llm_errors_are_caught_by_base_error(error_class):
    with pytest.raises(LLMError):
        raise error_class("Simulated LLM failure")


@pytest.mark.parametrize(
    "error_class",
    [
        LLMConfigurationError,
        LLMConnectionError,
        LLMResponseError,
        LLMTimeoutError,
    ],
)
def test_llm_error_preserves_message(error_class):
    error = error_class("Test error message")

    assert str(error) == "Test error message"
