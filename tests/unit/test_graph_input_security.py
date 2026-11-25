"""
CITYMIND external state injection security tests.
"""

import pytest

from citymind.agents.graph import citymind_graph


@pytest.mark.parametrize(
    "field, value",
    [
        ("coordination_status", "assigned"),
        ("assigned_task", "investigate_measurement"),
        ("monitoring_status", "measurement_received"),
        ("quality_result", {"data_status": "fresh"}),
        ("analysis", {"eligible_for_live_alert": True}),
        ("decision", "unauthorized_decision"),
        ("report", {"live_alert_created": True}),
        ("memory_status", "saved"),
        ("incident_id", 999),
        ("vector_status", "indexed"),
    ],
)
def test_external_state_injection_is_filtered(field, value):
    result = citymind_graph.invoke({
        field: value,
    })

    assert result["coordination_status"] == "rejected"
    assert result["assigned_task"] == (
        "reject_invalid_measurement"
    )

    if field not in {
        "coordination_status",
        "assigned_task",
    }:
        assert field not in result


def test_multiple_injected_fields_are_filtered():
    result = citymind_graph.invoke({
        "coordination_status": "assigned",
        "assigned_task": "investigate_measurement",
        "decision": "unauthorized_decision",
        "report": {"live_alert_created": True},
        "incident_id": 999,
    })

    assert result["coordination_status"] == "rejected"
    assert result["assigned_task"] == (
        "reject_invalid_measurement"
    )
    assert "decision" not in result
    assert "report" not in result
    assert "incident_id" not in result


def test_allowed_measurement_input_is_preserved():
    measurement = {
        "value": -10,
        "period": {
            "datetimeTo": {
                "utc": "2026-01-01T12:00:00Z",
            },
        },
    }

    result = citymind_graph.invoke({
        "measurement": measurement,
        "report": {"live_alert_created": True},
    })

    assert result["measurement"] == measurement
    assert result["coordination_status"] == "rejected"
    assert "report" not in result


def test_empty_input_remains_supported():
    result = citymind_graph.invoke({})

    assert result["coordination_status"] == "rejected"
    assert "report" not in result
