"""Regression tests for Gemini API version compatibility."""

from unittest.mock import patch

from citymind.llm.config import LLMConfig
from citymind.llm.gemini_provider import GeminiProvider


def test_gemini_provider_uses_explicit_v1beta_api_version():
    config = LLMConfig(
        enabled=True,
        provider="gemini",
        model="gemini-3.1-flash-lite",
        api_key="citymind-test-fake-key",
    )

    with patch(
        "citymind.llm.gemini_provider.genai.Client"
    ) as mock_client:

        with GeminiProvider(config):
            mock_client.assert_called_once()

            kwargs = mock_client.call_args.kwargs

            assert kwargs["vertexai"] is False

            options = kwargs["http_options"]

            assert options.api_version == "v1beta"

        mock_client.return_value.close.assert_called_once()
