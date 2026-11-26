"""Memory node numeric overflow integration tests."""

import pytest

from citymind.agents.memory_node import memory_node
from citymind.memory.database import get_incident


def build_state(value):
    return {
        "analysis": {
            "value": value,
            "data_status": "stale",
            "severity": "high",
        },
        "measurement": {
            "period": {
                "datetimeTo": {
                    "utc": "2026-01-01T12:00:00Z",
                },
            },
        },
        "decision": "historical_analysis_only",
    }


@pytest.mark.parametrize("value", [2**63, 10**100])
def test_memory_saves_large_finite_value(tmp_path, value):
    db_path = tmp_path / "memory.db"

    result = memory_node(build_state(value), db_path)

    assert result["memory_status"] == "saved"
    assert result["incident_id"] is not None

    incident = get_incident(result["incident_id"], db_path)

    assert incident is not None
    assert incident["measurement_value"] == float(value)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_memory_rejects_unrepresentable_value(tmp_path, value):
    db_path = tmp_path / "memory.db"

    result = memory_node(build_state(value), db_path)

    assert result == {
        "memory_status": "unavailable",
        "incident_id": None,
    }
    assert not db_path.exists()
