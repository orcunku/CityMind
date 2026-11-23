"""
Tests for CITYMIND SQLite recent incident listing.
"""

import pytest

from citymind.memory.database import (
    list_recent_incidents,
    save_incident,
)

def add_incident(db_path, value):
    return save_incident(
        measurement_value=value,
        measured_at="2022-10-25T12:00:00Z",
        data_status="stale",
        severity="historical",
        decision="historical_analysis_only",
        db_path=db_path,
    )

def test_list_recent_incidents_returns_newest_first(tmp_path):
    db_path = tmp_path / "incidents.db"

    for value in (12.0, 25.0, 40.0):
        add_incident(db_path, value)

    incidents = list_recent_incidents(
        limit=2,
        db_path=db_path,
    )

    assert len(incidents) == 2
    assert [
        incident["measurement_value"]
        for incident in incidents
    ] == [40.0, 25.0]

def test_list_recent_incidents_returns_all_available(tmp_path):
    db_path = tmp_path / "incidents.db"

    add_incident(db_path, 12.0)
    add_incident(db_path, 25.0)

    incidents = list_recent_incidents(
        limit=10,
        db_path=db_path,
    )

    assert len(incidents) == 2
    assert incidents[0]["id"] > incidents[1]["id"]

def test_list_recent_incidents_missing_database(tmp_path):
    db_path = tmp_path / "missing.db"

    assert list_recent_incidents(db_path=db_path) == []
    assert not db_path.exists()

@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 101, True, 1.5, "10", None],
)
def test_list_recent_incidents_rejects_invalid_limit(
    tmp_path,
    invalid_limit,
):
    db_path = tmp_path / "incidents.db"

    with pytest.raises(ValueError, match="Invalid incident limit"):
        list_recent_incidents(
            limit=invalid_limit,
            db_path=db_path,
        )

    assert not db_path.exists()
