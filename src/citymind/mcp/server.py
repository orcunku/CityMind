"""
CITYMIND MCP Server.

Exposes environmental monitoring tools
through the Model Context Protocol.
"""

from mcp.server.fastmcp import FastMCP

from citymind.detection.anomaly import (
    detect_pm25_anomaly,
)
from citymind.tools.measurement_quality import (
    measurement_quality_tool,
)


mcp = FastMCP("CityMind")


def build_measurement(
    value: float,
    measured_at: str,
) -> dict:
    """
    Build a measurement dictionary compatible
    with CityMind's ingestion layer.
    """
    return {
        "value": value,
        "period": {
            "datetimeTo": {
                "utc": measured_at,
            }
        },
    }


@mcp.tool()
def analyze_pm25(
    value: float,
    measured_at: str,
) -> dict:
    """
    Analyze a PM2.5 measurement.

    Returns severity, freshness and alert eligibility.

    Does not create or send live alerts.
    """
    measurement = build_measurement(
        value,
        measured_at,
    )
    return detect_pm25_anomaly(measurement)


@mcp.tool()
def check_measurement_quality(
    value: float,
    measured_at: str,
) -> dict:
    """
    Check measurement freshness.

    Returns the measurement quality tool result.

    Does not create or send live alerts.
    """
    measurement = build_measurement(
        value,
        measured_at,
    )
    return measurement_quality_tool(measurement)


if __name__ == "__main__":
    mcp.run(transport="stdio")
