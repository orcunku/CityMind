"""
CITYMIND provider-independent LLM interface.
"""

from types import TracebackType
from typing import Protocol, Self, runtime_checkable


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

    @property
    def closed(self) -> bool:
        """Return whether the provider is closed."""
        ...

    def close(self) -> None:
        """Release provider resources."""
        ...

    def __enter__(self) -> Self:
        """Enter the provider context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Release resources without suppressing errors."""
        ...
