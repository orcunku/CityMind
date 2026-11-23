"""
CITYMIND LangGraph SQLite memory node.

Persists successful analysis results without
creating or modifying live alerts.
"""

import sqlite3
from pathlib import Path

from citymind.memory.database import save_incident


def memory_node(
    state: dict,
    db_path: str | Path,
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

    return {
        "memory_status": "saved",
        "incident_id": incident_id,
    }
