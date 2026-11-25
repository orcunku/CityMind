"""Secure Groq LLM provider."""

import groq

from citymind.llm.config import LLMConfig, validate_llm_config
from citymind.llm.errors import (
    LLMConfigurationError,
    LLMConnectionError,
)


class GroqProvider:
    def __init__(self, config: LLMConfig):
        validate_llm_config(config)

        if not config.enabled:
            raise LLMConfigurationError(
                "LLM operations are disabled"
            )

        if config.provider != "groq":
            raise LLMConfigurationError(
                "Groq provider configuration required"
            )

        self._config = config
        self._closed = False

        try:
            self._client = groq.Groq(
                api_key=config.api_key,
                base_url="https://api.groq.com",
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        except Exception:
            raise LLMConnectionError(
                "Unable to initialize Groq client"
            ) from None

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            self._client.close()
        except Exception:
            raise LLMConnectionError(
                "Unable to close Groq client"
            ) from None

    def __enter__(self):
        if self._closed:
            raise LLMConfigurationError(
                "Groq provider is closed"
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except LLMConnectionError:
                pass
        return False

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        from citymind.llm.errors import (
            LLMResponseError,
            LLMTimeoutError,
            LLMRateLimitError,
        )

        if self._closed:
            raise LLMConfigurationError(
                "Groq provider is closed"
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

        messages = []

        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=self._config.max_output_tokens,
                n=1,
                stream=False,
            )
        except groq.APITimeoutError:
            raise LLMTimeoutError(
                "Groq request timed out"
            ) from None
        except groq.RateLimitError:
            raise LLMRateLimitError(
                "Groq rate limit or quota exceeded"
            ) from None
        except groq.APIStatusError as exc:
            if exc.status_code in {408, 504}:
                raise LLMTimeoutError(
                    "Groq request timed out"
                ) from None
            raise LLMConnectionError(
                "Groq API request failed"
            ) from None
        except Exception:
            raise LLMConnectionError(
                "Unable to complete Groq request"
            ) from None

        try:
            choices = response.choices

            if not choices or len(choices) != 1:
                raise LLMResponseError(
                    "Groq returned an invalid choice count"
                )

            choice = choices[0]

            if choice.finish_reason != "stop":
                raise LLMResponseError(
                    "Groq response did not complete normally"
                )

            text = choice.message.content

        except LLMResponseError:
            raise
        except Exception:
            raise LLMResponseError(
                "Unable to read Groq response"
            ) from None

        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(
                "Groq returned an empty response"
            )

        return text.strip()
