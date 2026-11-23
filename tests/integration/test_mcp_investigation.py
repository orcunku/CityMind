"""
Integration tests for the MCP-backed
CityMind investigation node.
"""

from datetime import datetime, timedelta, timezone

import pytest

from citymind.agents.mcp_investigation import (
    mcp_investigation_node,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_measurement(age_hours: int) -> dict:
    measured_at = (
        datetime.now(timezone.utc)
        - timedelta(hours=age_hours)
    ).isoformat()

    return {
        "value": 40.0,
        "period": {
            "datetimeTo": {
                "utc": measured_at,
            }
        },
    }


@pytest.mark.anyio
async def test_mcp_investigation_fresh():
    result = await mcp_investigation_node({
        "measurement": make_measurement(2),
    })

    quality = result["quality_result"]
    analysis = result["analysis"]

    assert quality["tool_name"] == "measurement_quality"
    assert quality["data_status"] == "fresh"
    assert quality["is_fresh"] is True

    assert analysis["severity"] == "high"
    assert analysis["data_status"] == "fresh"
    assert analysis["eligible_for_live_alert"] is True


@pytest.mark.anyio
async def test_mcp_investigation_stale():
    result = await mcp_investigation_node({
        "measurement": make_measurement(48),
    })

    quality = result["quality_result"]
    analysis = result["analysis"]

    assert quality["is_fresh"] is False
    assert quality["data_status"] != "fresh"

    assert analysis["data_status"] != "fresh"
    assert analysis["eligible_for_live_alert"] is False
