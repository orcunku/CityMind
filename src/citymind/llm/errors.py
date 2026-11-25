"""
CITYMIND LLM exception hierarchy.
"""


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMConfigurationError(LLMError):
    """Invalid or missing LLM configuration."""


class LLMConnectionError(LLMError):
    """Unable to communicate with the LLM provider."""


class LLMResponseError(LLMError):
    """Invalid or unusable LLM response."""


class LLMTimeoutError(LLMError):
    """LLM operation exceeded its time limit."""
