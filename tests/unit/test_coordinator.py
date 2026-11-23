"""
Unit tests for CITYMIND Coordinator Agent.
"""

import pytest

from citymind.agents.coordinator import coordinator_node


def test_coordinator_assigns_measurement():
    result = coordinator_node({
        "measurement": {
            "value": 25.0,
            "period": {
                "datetimeTo": {
                    "utc": "2022-10-25T12:00:00Z",
                },
            },
        },
    })

    assert result == {
        "coordination_status": "assigned",
        "assigned_task": "investigate_measurement",
    }


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
def test_coordinator_rejects_invalid_input(measurement):
    result = coordinator_node({
        "measurement": measurement,
    })

    assert result == {
        "coordination_status": "rejected",
        "assigned_task": "reject_invalid_measurement",
    }


def test_coordinator_rejects_missing_measurement():
    result = coordinator_node({})

    assert result["coordination_status"] == "rejected"
    assert result["assigned_task"] == (
        "reject_invalid_measurement"
    )


def test_coordinator_does_not_create_alert():
    result = coordinator_node({
        "measurement": {"value": 85.0},
    })

    assert "live_alert_created" not in result
    assert "report" not in result
    assert "decision" not in result


def test_coordinator_does_not_modify_input():
    state = {
        "measurement": {"value": 25.0},
    }

    original_measurement = state["measurement"].copy()

    coordinator_node(state)

    assert state["measurement"] == original_measurement



from citymind.agents.coordinator import route_after_coordination


def test_route_accepts_assigned_investigation():
    state = {
        "coordination_status": "assigned",
        "assigned_task": "investigate_measurement",
    }

    assert route_after_coordination(state) == "investigate"


def test_route_rejects_invalid_measurement():
    state = {
        "coordination_status": "rejected",
        "assigned_task": "reject_invalid_measurement",
    }

    assert route_after_coordination(state) == "reject"


def test_route_rejects_missing_state():
    assert route_after_coordination({}) == "reject"


def test_route_rejects_unknown_task():
    state = {
        "coordination_status": "assigned",
        "assigned_task": "unknown_task",
    }

    assert route_after_coordination(state) == "reject"


def test_route_rejects_inconsistent_status():
    state = {
        "coordination_status": "rejected",
        "assigned_task": "investigate_measurement",
    }

    assert route_after_coordination(state) == "reject"
