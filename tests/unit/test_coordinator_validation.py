"""
Coordinator measurement structure validation tests.
"""

import pytest

from citymind.agents.coordinator import (
    coordinator_node,
    is_valid_measurement_structure,
)


def make_measurement(value=25.0, timestamp="2022-10-25T12:00:00Z"):
    return {
        "value": value,
        "period": {
            "datetimeTo": {
                "utc": timestamp,
            },
        },
    }


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "25.0",
        None,
    ],
)
def test_invalid_pm25_values_are_rejected(value):
    measurement = make_measurement(value=value)

    assert is_valid_measurement_structure(measurement) is False

    result = coordinator_node({
        "measurement": measurement,
    })

    assert result["coordination_status"] == "rejected"


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "",
        "invalid-date",
        "2022-10-25T12:00:00",
        "2022-99-99T12:00:00Z",
        12345,
    ],
)
def test_invalid_timestamps_are_rejected(timestamp):
    measurement = make_measurement(timestamp=timestamp)

    assert is_valid_measurement_structure(measurement) is False

    result = coordinator_node({
        "measurement": measurement,
    })

    assert result["coordination_status"] == "rejected"


@pytest.mark.parametrize(
    "measurement",
    [
        {"value": 25.0},
        {"value": 25.0, "period": None},
        {"value": 25.0, "period": {}},
        {
            "value": 25.0,
            "period": {"datetimeTo": None},
        },
    ],
)
def test_missing_measurement_fields_are_rejected(measurement):
    assert is_valid_measurement_structure(measurement) is False


def test_historical_measurement_is_accepted():
    measurement = make_measurement()

    assert is_valid_measurement_structure(measurement) is True

    result = coordinator_node({
        "measurement": measurement,
    })

    assert result == {
        "coordination_status": "assigned",
        "assigned_task": "investigate_measurement",
    }


def test_zero_pm25_is_valid():
    measurement = make_measurement(value=0.0)

    assert is_valid_measurement_structure(measurement) is True
