"""
CITYMIND measurement quality tool.

Checks whether a measurement is recent enough
to be considered for live incident evaluation.
"""

from datetime import datetime

from citymind.ingestion.data_quality import (
    check_measurement_freshness,
)


def measurement_quality_tool(
    measurement: dict,
    now: datetime | None = None,
) -> dict:
    """
    Validate the timestamp and freshness
    of a PM2.5 measurement.

    This tool does not create alerts.
    """

    quality = check_measurement_freshness(
        measurement,
        now=now,
    )

    return {
        "tool_name": "measurement_quality",
        "data_status": quality["status"],
        "is_fresh": quality["is_fresh"],
        "eligible_for_live_evaluation": quality["is_fresh"],
        "measured_at": quality["measured_at"],
        "age_hours": quality["age_hours"],
    }
