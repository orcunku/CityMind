"""
CITYMIND provider-independent LLM interface.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """
    Common interface for supported LLM providers.

    Providers generate text but do not directly
    execute actions or modify incident memory.
    """

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a text response."""
        ...
