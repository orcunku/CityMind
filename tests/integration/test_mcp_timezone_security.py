"""
Verify MCP timestamp comparisons handle
equivalent timezone representations.
"""

from datetime import datetime, timedelta, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_accepts_equivalent_timezone_timestamp(
    monkeypatch,
):
    measured_at_utc = (
        datetime.now(timezone.utc)
        - timedelta(hours=2)
    )

    measured_at_izmir = measured_at_utc.astimezone(
        timezone(timedelta(hours=3))
    )

    async def fake_mcp_tool(tool_name, arguments):
        assert tool_name == "check_measurement_quality"
        assert arguments["measured_at"] == (
            measured_at_utc.isoformat()
        )

        return {
            "tool_name": "measurement_quality",
            "is_fresh": True,
            "data_status": "fresh",
            "eligible_for_live_evaluation": True,
            "measured_at": measured_at_izmir.isoformat(),
            "age_hours": 2.0,
        }

    monkeypatch.setattr(
        investigation_module,
        "call_citymind_mcp_tool",
        fake_mcp_tool,
    )

    measurement = {
        "value": 40.0,
        "period": {
            "datetimeTo": {
                "utc": measured_at_utc.isoformat(),
            }
        },
    }

    result = await investigation_module.mcp_investigation_node({
        "measurement": measurement,
    })

    assert result["quality_result"]["is_fresh"] is True
    assert result["analysis"]["data_status"] == "fresh"
    assert result["analysis"]["severity"] == "high"
