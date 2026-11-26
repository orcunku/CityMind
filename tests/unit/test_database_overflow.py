"""Regression tests for SQLite numeric overflow."""

import pytest

from citymind.memory.database import (
    get_incident,
    save_incident,
)


@pytest.mark.parametrize(
    "value",
    [2**63, 10**100],
)
def test_large_finite_measurement(tmp_path, value):
    db_path = tmp_path / "overflow.db"

    incident_id = save_incident(
        measurement_value=value,
        measured_at="2026-01-01T12:00:00Z",
        data_status="stale",
        severity="high",
        decision="historical_analysis_only",
        db_path=db_path,
    )

    incident = get_incident(incident_id, db_path)

    assert incident is not None
    assert incident["measurement_value"] == float(value)


@pytest.mark.parametrize(
    "value",
    [10**400, -(10**400)],
)
def test_unrepresentable_measurement(tmp_path, value):
    db_path = tmp_path / "overflow.db"

    with pytest.raises(
        ValueError,
        match="^Invalid measurement value$",
    ):
        save_incident(
            measurement_value=value,
            measured_at="2026-01-01T12:00:00Z",
            data_status="stale",
            severity="high",
            decision="historical_analysis_only",
            db_path=db_path,
        )

    assert not db_path.exists()


@pytest.mark.parametrize(
    "incident_id",
    [2**63, 10**100],
)
def test_incident_id_overflow(tmp_path, incident_id):
    db_path = tmp_path / "overflow.db"

    save_incident(
        measurement_value=42.0,
        measured_at="2026-01-01T12:00:00Z",
        data_status="stale",
        severity="high",
        decision="historical_analysis_only",
        db_path=db_path,
    )

    with pytest.raises(
        ValueError,
        match="^Invalid incident ID$",
    ):
        get_incident(incident_id, db_path)


def test_maximum_valid_incident_id(tmp_path):
    db_path = tmp_path / "boundary.db"

    save_incident(
        measurement_value=42.0,
        measured_at="2026-01-01T12:00:00Z",
        data_status="stale",
        severity="high",
        decision="historical_analysis_only",
        db_path=db_path,
    )

    assert get_incident(2**63 - 1, db_path) is None


@pytest.mark.parametrize(
    "incident_id",
    [2**63, 10**100],
)
def test_id_overflow_without_database(tmp_path, incident_id):
    db_path = tmp_path / "missing.db"

    with pytest.raises(
        ValueError,
        match="^Invalid incident ID$",
    ):
        get_incident(incident_id, db_path)

    assert not db_path.exists()
