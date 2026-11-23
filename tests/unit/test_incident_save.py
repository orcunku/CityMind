"""
Tests for CITYMIND SQLite incident persistence.
"""

import sqlite3

import pytest

from citymind.memory.database import save_incident

def test_save_incident_persists_record(tmp_path):
    db_path = tmp_path / "incidents.db"

    incident_id = save_incident(
        measurement_value=34.19,
        measured_at="2022-10-25T12:00:00Z",
        data_status="stale",
        severity="high",
        decision="historical_analysis_only",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT measurement_value, data_status, decision
            FROM incidents
            WHERE id = ?
            """,
            (incident_id,),
        ).fetchone()

    assert incident_id == 1
    assert row == (
        34.19,
        "stale",
        "historical_analysis_only",
    )

def test_save_incident_assigns_unique_ids(tmp_path):
    db_path = tmp_path / "incidents.db"

    def save(value):
        return save_incident(
            measurement_value=value,
            measured_at="2022-10-25T12:00:00Z",
            data_status="stale",
            severity="high",
            decision="historical_analysis_only",
            db_path=db_path,
        )

    first_id = save(34.19)
    second_id = save(40.0)

    assert first_id != second_id

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

    assert count == 2

@pytest.mark.parametrize(
    "invalid_value",
    [-1, float("nan"), float("inf"), True, "invalid"],
)
def test_save_incident_rejects_invalid_values(
    tmp_path,
    invalid_value,
):
    db_path = tmp_path / "incidents.db"

    with pytest.raises(ValueError, match="Invalid measurement"):
        save_incident(
            measurement_value=invalid_value,
            measured_at="2022-10-25T12:00:00Z",
            data_status="stale",
            severity="high",
            decision="historical_analysis_only",
            db_path=db_path,
        )

    assert not db_path.exists()
