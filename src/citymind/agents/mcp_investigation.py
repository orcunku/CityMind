"""
MCP-backed investigation node for CITYMIND.

Validates that MCP quality results correspond
to the measurement being investigated.
"""

from datetime import datetime, timezone
from typing import Any

from citymind.detection.anomaly import (
    detect_pm25_anomaly,
    validate_quality_result,
)
from citymind.mcp.client import call_citymind_mcp_tool
from citymind.mcp.errors import MCPInvalidResponseError


def parse_measurement_timestamp(value: str) -> datetime:
    """
    Parse a timezone-aware ISO 8601 timestamp.
    """
    if not isinstance(value, str):
        raise ValueError("Invalid measurement timestamp")

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "Invalid measurement timestamp"
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(
            "Measurement timestamp must include timezone"
        )

    return parsed.astimezone(timezone.utc)


async def mcp_investigation_node(
    state: dict[str, Any],
) -> dict:
    """
    Investigate a measurement using MCP quality data.
    """

    measurement = state.get("measurement")

    if not isinstance(measurement, dict):
        raise ValueError(
            "A valid measurement dictionary is required"
        )

    try:
        value = measurement["value"]
        measured_at = measurement["period"]["datetimeTo"]["utc"]
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

    if not isinstance(quality, dict):
        raise MCPInvalidResponseError(
            "Invalid MCP quality response"
        )

    expected_timestamp = parse_measurement_timestamp(
        measured_at
    )

    try:
        received_timestamp = parse_measurement_timestamp(
            quality.get("measured_at")
        )
    except ValueError as exc:
        raise MCPInvalidResponseError(
            "Invalid MCP measurement timestamp"
        ) from exc

    if received_timestamp != expected_timestamp:
        raise MCPInvalidResponseError(
            "MCP measurement timestamp mismatch"
        )

    try:
        validate_quality_result(quality)
    except ValueError as exc:
        raise MCPInvalidResponseError(
            "Invalid MCP quality data"
        ) from exc

    analysis = detect_pm25_anomaly(
        measurement,
        quality=quality,
    )

    return {
        "quality_result": quality,
        "analysis": analysis,
    }
