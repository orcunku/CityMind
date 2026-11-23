"""
CITYMIND MCP Server.

Exposes safe environmental analysis tools
through the Model Context Protocol.
"""

from mcp.server.fastmcp import FastMCP

from citymind.detection.anomaly import (
    detect_pm25_anomaly,
)


mcp = FastMCP("CityMind")


@mcp.tool()
def analyze_pm25(
    value: float,
    measured_at: str,
) -> dict:
    """
    Analyze a PM2.5 measurement.

    Args:
        value: PM2.5 concentration in µg/m³.
        measured_at: Measurement timestamp in ISO 8601
            format, including timezone information.

    Returns:
        Severity, freshness status and alert eligibility.

    This tool does not create or send live alerts.
    """

    measurement = {
        "value": value,
        "period": {
            "datetimeTo": {
                "utc": measured_at,
            }
        },
    }

    return detect_pm25_anomaly(measurement)


if __name__ == "__main__":
    mcp.run(transport="stdio")
