"""CITYMIND provider-independent embedding interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Common interface for semantic embedding providers."""

    @property
    def model_id(self) -> str:
        """Return the embedding model identity."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate a semantic search query embedding."""
        ...

    def embed_document(self, text: str) -> list[float]:
        """Generate a semantic document embedding."""
        ...
