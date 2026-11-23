"""
Integration tests for the CITYMIND MCP client bridge.
"""

from datetime import datetime, timedelta, timezone

import pytest

from citymind.mcp.client import call_citymind_mcp_tool


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_client_fresh_measurement():
    measured_at = (
        datetime.now(timezone.utc)
        - timedelta(hours=2)
    ).isoformat()

    result = await call_citymind_mcp_tool(
        "check_measurement_quality",
        {
            "value": 40.0,
            "measured_at": measured_at,
        },
    )

    assert result["tool_name"] == "measurement_quality"
    assert result["data_status"] == "fresh"
    assert result["is_fresh"] is True
    assert result["eligible_for_live_evaluation"] is True


@pytest.mark.anyio
async def test_mcp_client_stale_measurement():
    measured_at = (
        datetime.now(timezone.utc)
        - timedelta(hours=48)
    ).isoformat()

    result = await call_citymind_mcp_tool(
        "check_measurement_quality",
        {
            "value": 40.0,
            "measured_at": measured_at,
        },
    )

    assert result["tool_name"] == "measurement_quality"
    assert result["data_status"] != "fresh"
    assert result["is_fresh"] is False
    assert result["eligible_for_live_evaluation"] is False
