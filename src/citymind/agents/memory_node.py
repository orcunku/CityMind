"""
CITYMIND LangGraph SQLite memory node.

Persists successful analysis results without
creating or modifying live alerts.
"""

import sqlite3
from pathlib import Path

from citymind.memory.database import save_incident
from citymind.memory.indexer import index_sqlite_incident


def memory_node(
    state: dict,
    db_path: str | Path,
    chroma_path: str | Path | None = None,
) -> dict:
    """
    Persist an analyzed measurement safely.

    Returns memory status and incident ID.
    """
    analysis = state.get("analysis", {})
    measurement = state.get("measurement", {})
    decision = state.get("decision")

    if (
        not isinstance(analysis, dict)
        or not isinstance(measurement, dict)
        or analysis.get("data_status") == "unavailable"
        or decision == "investigation_unavailable"
    ):
        return {
            "memory_status": "skipped",
            "incident_id": None,
        }

    try:
        incident_id = save_incident(
            measurement_value=analysis["value"],
            measured_at=measurement["period"]["datetimeTo"]["utc"],
            data_status=analysis["data_status"],
            severity=analysis["severity"],
            decision=decision,
            db_path=db_path,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        sqlite3.Error,
        OSError,
    ):
        return {
            "memory_status": "unavailable",
            "incident_id": None,
        }

    result = {
        "memory_status": "saved",
        "incident_id": incident_id,
    }

    if chroma_path is not None:
        try:
            index_result = index_sqlite_incident(
                incident_id=incident_id,
                db_path=db_path,
                persist_directory=chroma_path,
            )
            result["vector_status"] = index_result["index_status"]
        except Exception:
            # Vector indexing must not invalidate SQLite persistence.
            result["vector_status"] = "unavailable"

    return result
