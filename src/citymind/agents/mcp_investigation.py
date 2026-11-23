"""
CITYMIND MCP-backed investigation node.

Uses the MCP protocol to obtain measurement
quality information before anomaly detection.

Does not create or send live alerts.
"""

from typing import Any

from citymind.detection.anomaly import detect_pm25_anomaly
from citymind.mcp.client import call_citymind_mcp_tool


async def mcp_investigation_node(
    state: dict[str, Any],
) -> dict:
    """
    Investigate a PM2.5 measurement using MCP.

    Reuse the MCP quality result for anomaly
    detection and return LangGraph state updates.
    """

    measurement = state.get("measurement")

    if not isinstance(measurement, dict):
        raise ValueError(
            "A valid measurement dictionary is required"
        )

    try:
        value = measurement["value"]
        measured_at = measurement[
            "period"
        ]["datetimeTo"]["utc"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Measurement value and timestamp are required"
        ) from exc

    quality = await call_citymind_mcp_tool(
        "check_measurement_quality",
        {
            "value": value,
            "measured_at": measured_at,
        },
    )

    analysis = detect_pm25_anomaly(
        measurement,
        quality=quality,
    )

    return {
        "quality_result": quality,
        "analysis": analysis,
    }
