"""
CITYMIND LLM configuration management.

LLM operations are disabled by default.
"""

import os
import math
from dataclasses import dataclass, field

from citymind.llm.errors import LLMConfigurationError

SUPPORTED_PROVIDERS = frozenset({
    "gemini",
    "groq",
})

@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    provider: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1024

def validate_llm_config(config: LLMConfig) -> None:
    """Validate configuration before provider initialization."""

    if not isinstance(config, LLMConfig):
        raise LLMConfigurationError(
            "Invalid LLM configuration object"
        )

    if type(config.enabled) is not bool:
        raise LLMConfigurationError(
            "LLM enabled flag must be boolean"
        )

    if (
        not isinstance(config.provider, str)
        or config.provider not in SUPPORTED_PROVIDERS
    ):
        raise LLMConfigurationError(
            "Unsupported LLM provider"
        )

    if not isinstance(config.model, str):
        raise LLMConfigurationError(
            "Invalid LLM model"
        )

    if not isinstance(config.api_key, str):
        raise LLMConfigurationError(
            "Invalid LLM API key"
        )

    if config.enabled and not config.model.strip():
        raise LLMConfigurationError(
            "LLM model is required"
        )

    if config.enabled and not config.api_key.strip():
        raise LLMConfigurationError(
            "LLM API key is required"
        )

    timeout = config.timeout_seconds

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= 120
        or not math.isfinite(timeout)
    ):
        raise LLMConfigurationError(
            "Invalid LLM timeout"
        )

    tokens = config.max_output_tokens

    if (
        type(tokens) is not int
        or not 1 <= tokens <= 4096
    ):
        raise LLMConfigurationError(
            "Invalid LLM output token limit"
        )


def load_llm_config() -> LLMConfig:
    """Load and validate LLM environment settings."""

    enabled_value = os.getenv(
        "LLM_ENABLED", "false"
    ).strip().lower()

    if enabled_value not in {"true", "false"}:
        raise LLMConfigurationError(
            "LLM_ENABLED must be true or false"
        )

    enabled = enabled_value == "true"

    provider = os.getenv(
        "LLM_PROVIDER", "gemini"
    ).strip().lower()

    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            "Unsupported LLM provider"
        )

    model = os.getenv(
        "LLM_MODEL", ""
    ).strip()

    key_variable = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }[provider]

    api_key = os.getenv(
        key_variable, ""
    ).strip()

    try:
        timeout_seconds = float(
            os.getenv("LLM_TIMEOUT_SECONDS", "30")
        )

        max_output_tokens = int(
            os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024")
        )
    except (ValueError, OverflowError):
        raise LLMConfigurationError(
            "Invalid LLM numeric configuration"
        ) from None

    if (
        not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 120
    ):
        raise LLMConfigurationError(
            "LLM timeout must be between 0 and 120 seconds"
        )

    if not 1 <= max_output_tokens <= 4096:
        raise LLMConfigurationError(
            "Invalid LLM output token limit"
        )

    if enabled and not model:
        raise LLMConfigurationError(
            "LLM_MODEL is required when LLM is enabled"
        )

    if enabled and not api_key:
        raise LLMConfigurationError(
            f"{key_variable} is required"
        )

    config = LLMConfig(
        enabled=enabled,
        provider=provider,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )

    validate_llm_config(config)
    return config
