"""
CITYMIND SQLite-to-ChromaDB incident indexer.

SQLite is the source of truth.
ChromaDB is a searchable secondary index.
"""

from pathlib import Path

from citymind.memory.database import get_incident
from citymind.memory.vector_store import save_incident_vector


def index_sqlite_incident(
    incident_id: int,
    db_path: str | Path,
    persist_directory: str | Path,
) -> dict:
    """
    Index an existing SQLite incident in ChromaDB.

    Does not modify SQLite records or create alerts.
    """

    incident = get_incident(
        incident_id,
        db_path=db_path,
    )

    if incident is None:
        return {
            "index_status": "not_found",
            "vector_id": None,
        }

    try:
        vector_id = save_incident_vector(
            incident_id=incident["id"],
            measurement_value=incident["measurement_value"],
            data_status=incident["data_status"],
            severity=incident["severity"],
            decision=incident["decision"],
            persist_directory=persist_directory,
        )
    except (ValueError, TypeError, OSError):
        return {
            "index_status": "unavailable",
            "vector_id": None,
        }

    return {
        "index_status": "indexed",
        "vector_id": vector_id,
    }
