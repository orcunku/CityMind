"""
CITYMIND Gemini LLM provider adapter.
"""

from google import genai
from google.genai import types, errors

from citymind.llm.config import LLMConfig
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)


class GeminiProvider:
    """Google Gemini implementation of LLMProvider."""

    def __init__(self, config: LLMConfig):
        if not config.enabled:
            raise LLMConfigurationError(
                "LLM operations are disabled"
            )

        if config.provider != "gemini":
            raise LLMConfigurationError(
                "Gemini provider configuration required"
            )

        if not config.api_key or not config.model:
            raise LLMConfigurationError(
                "Gemini API key and model are required"
            )

        self._config = config

        self._client = genai.Client(
            api_key=config.api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                timeout=max(
                    1,
                    int(config.timeout_seconds * 1000),
                ),
                retry_options=types.HttpRetryOptions(
                    attempts=1,
                ),
            ),
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:

        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMConfigurationError(
                "A non-empty prompt is required"
            )

        if system_prompt is not None and (
            not isinstance(system_prompt, str)
            or not system_prompt.strip()
        ):
            raise LLMConfigurationError(
                "Invalid system prompt"
            )

        try:
            response = self._client.models.generate_content(
                model=self._config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=(
                        self._config.max_output_tokens
                    ),
                    temperature=0.0,
                    automatic_function_calling=(
                        types.AutomaticFunctionCallingConfig(
                            disable=True,
                        )
                    ),
                ),
            )

        except TimeoutError:
            raise LLMTimeoutError(
                "Gemini request timed out"
            ) from None

        except errors.APIError as exc:
            if getattr(exc, "code", None) in {408, 504}:
                raise LLMTimeoutError(
                    "Gemini request timed out"
                ) from None

            raise LLMConnectionError(
                "Gemini API request failed"
            ) from None

        except Exception:
            raise LLMConnectionError(
                "Unable to complete Gemini request"
            ) from None

        try:
            text = response.text
        except Exception:
            raise LLMResponseError(
                "Unable to read Gemini response"
            ) from None

        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(
                "Gemini returned an empty response"
            )

        return text.strip()
