"""
Coordinator rejection and memory safety tests.
"""

import sqlite3

import pytest

from citymind.agents.graph import citymind_graph


@pytest.mark.parametrize(
    "measurement",
    [
        None,
        {},
        [],
        "",
        25.0,
    ],
)
def test_invalid_measurement_stops_before_memory(
    measurement,
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

    events = list(
        citymind_graph.stream({
            "measurement": measurement,
        })
    )

    visited = [
        node
        for event in events
        for node in event
    ]

    assert visited == ["coordinator"]

    result = events[0]["coordinator"]

    assert result["coordination_status"] == "rejected"
    assert result["assigned_task"] == (
        "reject_invalid_measurement"
    )

    assert not db_path.exists()
    assert not chroma_path.exists()


def test_missing_measurement_stops_safely(
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

    result = citymind_graph.invoke({})

    assert result["coordination_status"] == "rejected"

    assert "analysis" not in result
    assert "report" not in result
    assert "incident_id" not in result

    assert not db_path.exists()
    assert not chroma_path.exists()
