"""
Tests for CITYMIND SQLite incident retrieval.
"""

import pytest

from citymind.memory.database import (
    get_incident,
    save_incident,
)


def test_get_incident_returns_saved_record(tmp_path):
    db_path = tmp_path / "incidents.db"

    incident_id = save_incident(
        measurement_value=34.19,
        measured_at="2022-10-25T12:00:00Z",
        data_status="stale",
        severity="high",
        decision="historical_analysis_only",
        db_path=db_path,
    )

    incident = get_incident(incident_id, db_path)

    assert incident is not None
    assert incident["id"] == incident_id
    assert incident["measurement_value"] == 34.19
    assert incident["measured_at"] == "2022-10-25T12:00:00Z"
    assert incident["data_status"] == "stale"
    assert incident["severity"] == "high"
    assert incident["decision"] == "historical_analysis_only"
    assert incident["created_at"] is not None


def test_get_incident_returns_none_for_missing_id(tmp_path):
    db_path = tmp_path / "incidents.db"

    save_incident(
        measurement_value=20.0,
        measured_at="2022-10-25T12:00:00Z",
        data_status="stale",
        severity="moderate",
        decision="historical_analysis_only",
        db_path=db_path,
    )

    assert get_incident(999, db_path) is None


def test_get_incident_does_not_create_missing_database(tmp_path):
    db_path = tmp_path / "missing.db"

    assert get_incident(1, db_path) is None
    assert not db_path.exists()


@pytest.mark.parametrize(
    "invalid_id",
    [0, -1, True, 1.5, "1", None],
)
def test_get_incident_rejects_invalid_ids(tmp_path, invalid_id):
    db_path = tmp_path / "incidents.db"

    with pytest.raises(ValueError, match="Invalid incident ID"):
        get_incident(invalid_id, db_path)

    assert not db_path.exists()
