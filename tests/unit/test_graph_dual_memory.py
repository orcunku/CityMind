"""
Tests for CITYMIND LangGraph dual memory integration.
"""

import sqlite3
from unittest.mock import patch

from citymind.agents.graph import persist_memory_node
from citymind.memory.database import get_incident
from citymind.memory.vector_store import get_incident_collection

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

def test_langgraph_writes_to_both_memories(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "incidents.db"
    chroma_path = tmp_path / "chroma"

    monkeypatch.setenv(
        "CITYMIND_DB_PATH",
        str(db_path),
    )
    monkeypatch.setenv(
        "CITYMIND_CHROMA_PATH",
        str(chroma_path),
    )

    result = persist_memory_node(historical_state())

    assert result["memory_status"] == "saved"
    assert result["vector_status"] == "indexed"

    incident_id = result["incident_id"]

    sqlite_record = get_incident(
        incident_id,
        db_path=db_path,
    )

    assert sqlite_record is not None
    assert sqlite_record["data_status"] == "stale"

    collection = get_incident_collection(chroma_path)

    vector_record = collection.get(
        ids=[f"incident-{incident_id}"],
        include=["metadatas"],
    )

    assert vector_record["ids"] == [
        f"incident-{incident_id}"
    ]

    assert (
        vector_record["metadatas"][0]["incident_id"]
        == incident_id
    )

def test_sqlite_survives_vector_index_failure(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "incidents.db"
    chroma_path = tmp_path / "chroma"

    monkeypatch.setenv(
        "CITYMIND_DB_PATH",
        str(db_path),
    )
    monkeypatch.setenv(
        "CITYMIND_CHROMA_PATH",
        str(chroma_path),
    )

    with patch(
        "citymind.agents.memory_node.index_sqlite_incident",
        side_effect=RuntimeError(
            "simulated vector index failure"
        ),
    ):
        result = persist_memory_node(historical_state())

    assert result["memory_status"] == "saved"
    assert result["vector_status"] == "unavailable"

    sqlite_record = get_incident(
        result["incident_id"],
        db_path=db_path,
    )

    assert sqlite_record is not None
    assert sqlite_record["decision"] == (
        "historical_analysis_only"
    )
