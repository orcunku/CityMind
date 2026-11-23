"""
CITYMIND Coordinator Agent.

Assigns a deterministic task based on
the incoming measurement structure.
"""

from typing import Any
import math

from citymind.ingestion.data_quality import parse_utc_datetime


def is_valid_measurement_structure(measurement: Any) -> bool:
    """
    Validate the minimum PM2.5 measurement structure.

    Historical measurements remain valid inputs.
    Freshness is evaluated by the investigation agent.
    """
    if not isinstance(measurement, dict):
        return False

    value = measurement.get("value")

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return False

    period = measurement.get("period")
    if not isinstance(period, dict):
        return False

    datetime_to = period.get("datetimeTo")
    if not isinstance(datetime_to, dict):
        return False

    timestamp = datetime_to.get("utc")
    if not isinstance(timestamp, str) or not timestamp.strip():
        return False

    try:
        parse_utc_datetime(timestamp)
    except (ValueError, TypeError, OverflowError):
        return False

    return True


def coordinator_node(
    state: dict[str, Any],
) -> dict[str, str]:
    """
    Determine whether a measurement can
    proceed to investigation.

    This node does not create live alerts.
    """

    measurement = state.get("measurement")

    if not is_valid_measurement_structure(measurement):
        return {
            "coordination_status": "rejected",
            "assigned_task": "reject_invalid_measurement",
        }

    return {
        "coordination_status": "assigned",
        "assigned_task": "investigate_measurement",
    }



def route_after_coordination(
    state: dict[str, Any],
) -> str:
    """
    Select the next workflow branch.

    Only the explicitly assigned investigation
    task is allowed to proceed.
    All other tasks are rejected safely.
    """

    if (
        state.get("coordination_status") == "assigned"
        and state.get("assigned_task")
        == "investigate_measurement"
    ):
        return "investigate"

    return "reject"
