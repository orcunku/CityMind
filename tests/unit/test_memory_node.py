"""
Tests for CITYMIND LangGraph SQLite memory node.
"""

import sqlite3
from unittest.mock import patch

from citymind.agents.memory_node import memory_node
from citymind.memory.database import get_incident


def historical_state():
    return {
        "measurement": {
            "period": {
                "datetimeTo": {
                    "utc": "2022-10-25T12:00:00Z",
                },
            },
        },
        "analysis": {
            "value": 34.19,
            "data_status": "stale",
            "severity": "high",
        },
        "decision": "historical_analysis_only",
    }


def test_memory_node_saves_historical_incident(tmp_path):
    db_path = tmp_path / "incidents.db"

    result = memory_node(
        historical_state(),
        db_path=db_path,
    )

    assert result["memory_status"] == "saved"
    assert isinstance(result["incident_id"], int)

    incident = get_incident(
        result["incident_id"],
        db_path=db_path,
    )

    assert incident["measurement_value"] == 34.19
    assert incident["data_status"] == "stale"
    assert incident["decision"] == "historical_analysis_only"


def test_memory_node_skips_unavailable_investigation(tmp_path):
    db_path = tmp_path / "incidents.db"

    state = {
        "measurement": {
            "period": {
                "datetimeTo": {
                    "utc": "2022-10-25T12:00:00Z",
                },
            },
        },
        "analysis": {
            "data_status": "unavailable",
        },
        "decision": "investigation_unavailable",
    }

    result = memory_node(state, db_path=db_path)

    assert result == {
        "memory_status": "skipped",
        "incident_id": None,
    }
    assert not db_path.exists()


def test_memory_node_handles_missing_analysis_fields(tmp_path):
    db_path = tmp_path / "incidents.db"

    state = {
        "measurement": {
            "period": {
                "datetimeTo": {
                    "utc": "2022-10-25T12:00:00Z",
                },
            },
        },
        "analysis": {
            "data_status": "stale",
        },
        "decision": "historical_analysis_only",
    }

    result = memory_node(state, db_path=db_path)

    assert result == {
        "memory_status": "unavailable",
        "incident_id": None,
    }
    assert not db_path.exists()


def test_memory_node_handles_database_failure(tmp_path):
    db_path = tmp_path / "incidents.db"

    with patch(
        "citymind.agents.memory_node.save_incident",
        side_effect=sqlite3.OperationalError(
            "database unavailable"
        ),
    ):
        result = memory_node(
            historical_state(),
            db_path=db_path,
        )

    assert result == {
        "memory_status": "unavailable",
        "incident_id": None,
    }
