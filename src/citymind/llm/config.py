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

    return LLMConfig(
        enabled=enabled,
        provider=provider,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
    )
