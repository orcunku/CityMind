"""
CITYMIND Gemini LLM provider adapter.
"""

import httpx
import httpx2

from google import genai
from google.genai import types, errors

from citymind.llm.config import (
    LLMConfig,
    validate_llm_config,
)
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    LLMRateLimitError,
)


class GeminiProvider:
    """Google Gemini implementation of LLMProvider."""

    def __init__(self, config: LLMConfig):
        validate_llm_config(config)

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
        self._closed = False

        try:
            self._client = genai.Client(
                api_key=config.api_key,
                vertexai=False,
                http_options=types.HttpOptions(
                    api_version="v1beta",
                    timeout=max(
                        1,
                        int(config.timeout_seconds * 1000),
                    ),
                    retry_options=types.HttpRetryOptions(
                        attempts=1,
                    ),
                ),
            )

        except Exception:
            raise LLMConnectionError(
                "Unable to initialize Gemini client"
            ) from None

    @property
    def closed(self) -> bool:
        """Return whether the provider has been closed."""
        return self._closed

    def close(self) -> None:
        """Release Gemini client resources."""
        if self._closed:
            return

        self._closed = True

        try:
            self._client.close()
        except Exception:
            raise LLMConnectionError(
                "Unable to close Gemini client"
            ) from None

    def __enter__(self):
        """Enter the provider context."""
        if self._closed:
            raise LLMConfigurationError(
                "Gemini provider is closed"
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release resources when leaving the context."""
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except LLMConnectionError:
                # Preserve the original operation error.
                pass

        return False

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:

        if self._closed:
            raise LLMConfigurationError(
                "Gemini provider is closed"
            )

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

        except (
            TimeoutError,
            httpx.TimeoutException,
            httpx2.TimeoutException,
        ):
            raise LLMTimeoutError(
                "Gemini request timed out"
            ) from None

        except errors.APIError as exc:
            if getattr(exc, "code", None) == 429:
                raise LLMRateLimitError(
                    "Gemini rate limit or quota exceeded"
                ) from None

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
            candidates = response.candidates

            if not candidates or len(candidates) != 1:
                raise LLMResponseError(
                    "Gemini returned an invalid candidate count"
                )

            finish_reason = candidates[0].finish_reason

            if finish_reason != types.FinishReason.STOP:
                raise LLMResponseError(
                    "Gemini response did not complete normally"
                )

        except LLMResponseError:
            raise

        except Exception:
            raise LLMResponseError(
                "Unable to validate Gemini response"
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
