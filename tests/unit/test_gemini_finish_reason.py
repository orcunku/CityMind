"""
CITYMIND Gemini response completion validation.
"""

from unittest.mock import patch, MagicMock

import pytest

from google.genai import types

from citymind.llm.config import LLMConfig
from citymind.llm.gemini_provider import GeminiProvider
from citymind.llm.errors import LLMResponseError


@pytest.fixture
def provider_and_generate():
    config = LLMConfig(
        enabled=True,
        provider="gemini",
        model="mock-model",
        api_key="mock-key",
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as client:
        provider = GeminiProvider(config)
        generate = (
            client.return_value.models.generate_content
        )

        yield provider, generate


def test_completed_response_accepted(
    provider_and_generate,
):
    provider, generate = provider_and_generate

    generate.return_value = MagicMock(
        candidates=[
            MagicMock(
                finish_reason=types.FinishReason.STOP
            )
        ],
        text="Completed environmental analysis",
    )

    assert provider.generate("Analyze data") == (
        "Completed environmental analysis"
    )


@pytest.mark.parametrize(
    "reason",
    [
        reason
        for reason in types.FinishReason
        if reason != types.FinishReason.STOP
    ] + [None, "UNKNOWN_REASON"],
)
def test_incomplete_response_rejected(
    provider_and_generate, reason
):
    provider, generate = provider_and_generate

    generate.return_value = MagicMock(
        candidates=[
            MagicMock(finish_reason=reason)
        ],
        text="Partial environmental analysis",
    )

    with pytest.raises(LLMResponseError):
        provider.generate("Analyze data")


@pytest.mark.parametrize("candidates", [None, []])
def test_missing_candidates_rejected(
    provider_and_generate, candidates
):
    provider, generate = provider_and_generate

    generate.return_value = MagicMock(
        candidates=candidates,
        text="Unexpected response",
    )

    with pytest.raises(LLMResponseError):
        provider.generate("Analyze data")


def test_multiple_candidates_rejected(
    provider_and_generate,
):
    provider, generate = provider_and_generate

    generate.return_value = MagicMock(
        candidates=[
            MagicMock(
                finish_reason=types.FinishReason.STOP
            ),
            MagicMock(
                finish_reason=types.FinishReason.STOP
            ),
        ],
        text="Ambiguous response",
    )

    with pytest.raises(LLMResponseError):
        provider.generate("Analyze data")
