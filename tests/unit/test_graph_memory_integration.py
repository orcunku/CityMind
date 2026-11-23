"""
End-to-end tests for CITYMIND LangGraph SQLite memory.
"""

from citymind.agents.graph import citymind_graph
from citymind.memory.database import (
    get_incident,
    list_recent_incidents,
)

def historical_measurement():
    return {
        "value": 34.19,
        "period": {
            "datetimeTo": {
                "utc": "2022-10-25T12:00:00Z",
            },
        },
    }

def test_graph_persists_historical_incident(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "incidents.db"

    monkeypatch.setenv(
        "CITYMIND_DB_PATH",
        str(db_path),
    )

    result = citymind_graph.invoke({
        "measurement": historical_measurement(),
    })

    assert result["decision"] == (
        "historical_analysis_only"
    )
    assert result["analysis"]["data_status"] == "stale"
    assert result["report"]["live_alert_created"] is False

    assert result["memory_status"] == "saved"
    assert isinstance(result["incident_id"], int)

    incident = get_incident(
        result["incident_id"],
        db_path=db_path,
    )

    assert incident is not None
    assert incident["measurement_value"] == 34.19
    assert incident["data_status"] == "stale"
    assert incident["decision"] == (
        "historical_analysis_only"
    )
    assert incident["measured_at"] == (
        "2022-10-25T12:00:00Z"
    )

def test_graph_persists_multiple_incidents(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "incidents.db"

    monkeypatch.setenv(
        "CITYMIND_DB_PATH",
        str(db_path),
    )

    first = historical_measurement()

    second = historical_measurement()
    second["value"] = 42.0

    first_result = citymind_graph.invoke({
        "measurement": first,
    })

    second_result = citymind_graph.invoke({
        "measurement": second,
    })

    assert first_result["memory_status"] == "saved"
    assert second_result["memory_status"] == "saved"

    incidents = list_recent_incidents(
        limit=10,
        db_path=db_path,
    )

    assert len(incidents) == 2

    # Most recent incident comes first.
    assert incidents[0]["measurement_value"] == 42.0
    assert incidents[1]["measurement_value"] == 34.19

    # Historical data must never create live alerts.
    assert first_result["report"]["live_alert_created"] is False
    assert second_result["report"]["live_alert_created"] is False
