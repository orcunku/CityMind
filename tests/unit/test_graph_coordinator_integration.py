"""
CITYMIND Coordinator and LangGraph integration tests.
"""

from datetime import datetime, timedelta, timezone

import pytest

from citymind.agents.graph import citymind_graph


def make_measurement(age_hours: int) -> dict:
    measured_at = (
        datetime.now(timezone.utc)
        - timedelta(hours=age_hours)
    ).isoformat()

    return {
        "value": 25.0,
        "parameter": {
            "name": "pm25",
            "units": "µg/m³",
        },
        "period": {
            "datetimeTo": {
                "utc": measured_at,
            },
        },
    }


@pytest.mark.parametrize(
    "age_hours",
    [1, 100],
)
def test_coordinator_assigns_task_in_graph(age_hours):
    result = citymind_graph.invoke({
        "measurement": make_measurement(age_hours),
    })

    assert result["coordination_status"] == "assigned"
    assert result["assigned_task"] == (
        "investigate_measurement"
    )

    assert result["monitoring_status"] == (
        "measurement_received"
    )

    assert result["report"]["live_alert_created"] is False


def test_coordinator_is_first_graph_node():
    events = list(
        citymind_graph.stream({
            "measurement": make_measurement(100),
        })
    )

    assert list(events[0].keys()) == ["coordinator"]

    coordinator_result = events[0]["coordinator"]

    assert coordinator_result["coordination_status"] == (
        "assigned"
    )


def test_historical_measurement_remains_safe():
    result = citymind_graph.invoke({
        "measurement": make_measurement(100),
    })

    assert result["decision"] == (
        "historical_analysis_only"
    )

    assert result["report"]["live_alert_created"] is False
