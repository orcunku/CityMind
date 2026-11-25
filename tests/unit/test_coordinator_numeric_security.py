"""
CITYMIND coordinator numeric security tests.
"""

import pytest

from citymind.agents.coordinator import (
    coordinator_node,
    is_valid_measurement_structure,
)
from citymind.agents.graph import citymind_graph


def make_measurement(value):
    return {
        "value": value,
        "period": {
            "datetimeTo": {
                "utc": "2026-01-01T12:00:00Z",
            },
        },
    }


@pytest.mark.parametrize(
    "value",
    [
        10 ** 400,
        -(10 ** 400),
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
    ids=[
        "huge_positive_integer",
        "huge_negative_integer",
        "positive_infinity",
        "negative_infinity",
        "nan",
    ],
)
def test_extreme_numeric_input_is_rejected(value):
    measurement = make_measurement(value)

    assert is_valid_measurement_structure(
        measurement
    ) is False


def test_coordinator_rejects_huge_integer():
    result = coordinator_node({
        "measurement": make_measurement(10 ** 400),
    })

    assert result["coordination_status"] == "rejected"
    assert result["assigned_task"] == (
        "reject_invalid_measurement"
    )


def test_graph_rejects_huge_integer_safely():
    result = citymind_graph.invoke({
        "measurement": make_measurement(10 ** 400),
    })

    assert result["coordination_status"] == "rejected"
    assert "analysis" not in result
    assert "report" not in result
    assert "incident_id" not in result


@pytest.mark.parametrize("value", [0, 15, 25.5, 45.0])
def test_valid_numeric_inputs_remain_supported(value):
    assert is_valid_measurement_structure(
        make_measurement(value)
    ) is True
