"""Central factory for CITYMIND LLM providers."""

from citymind.llm.config import (
    LLMConfig,
    validate_llm_config,
)
from citymind.llm.errors import LLMConfigurationError
from citymind.llm.provider import LLMProvider


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    validate_llm_config(config)

    if not config.enabled:
        raise LLMConfigurationError(
            "LLM operations are disabled"
        )

    if config.provider == "groq":
        from citymind.llm.groq_provider import GroqProvider

        return GroqProvider(config)

    if config.provider == "gemini":
        from citymind.llm.gemini_provider import GeminiProvider

        return GeminiProvider(config)

    raise LLMConfigurationError(
        "Unsupported LLM provider"
    )
