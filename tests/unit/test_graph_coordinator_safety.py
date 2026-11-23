"""
End-to-end Coordinator safety routing tests.
"""

import pytest

from citymind.agents.graph import citymind_graph


def make_measurement(value, timestamp):
    return {
        "value": value,
        "period": {
            "datetimeTo": {
                "utc": timestamp,
            },
        },
    }


@pytest.mark.parametrize(
    "measurement",
    [
        make_measurement(-1.0, "2022-10-25T12:00:00Z"),
        make_measurement(float("nan"), "2022-10-25T12:00:00Z"),
        make_measurement(float("inf"), "2022-10-25T12:00:00Z"),
        make_measurement(25.0, "invalid-date"),
        make_measurement(25.0, "2022-10-25T12:00:00"),
        {"value": 25.0},
    ],
)
def test_invalid_measurement_never_reaches_analysis_or_memory(
    measurement,
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "incidents.db"
    chroma_path = tmp_path / "chroma"

    monkeypatch.setenv("CITYMIND_DB_PATH", str(db_path))
    monkeypatch.setenv("CITYMIND_CHROMA_PATH", str(chroma_path))

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


def test_historical_measurement_is_analyzed_safely(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "CITYMIND_DB_PATH",
        str(tmp_path / "incidents.db"),
    )
    monkeypatch.setenv(
        "CITYMIND_CHROMA_PATH",
        str(tmp_path / "chroma"),
    )

    result = citymind_graph.invoke({
        "measurement": make_measurement(
            45.0,
            "2022-10-25T12:00:00Z",
        ),
    })

    assert result["coordination_status"] == "assigned"
    assert result["analysis"]["data_status"] == "stale"
    assert result["decision"] == "historical_analysis_only"
    assert result["report"]["live_alert_created"] is False
