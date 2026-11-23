"""
Security: MCP quality results must belong
to the measurement being investigated.
"""

from datetime import datetime, timedelta, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_rejects_quality_for_different_timestamp(
    monkeypatch,
):
    now = datetime.now(timezone.utc)

    actual_timestamp = (
        now - timedelta(hours=48)
    ).isoformat()

    incorrect_timestamp = (
        now - timedelta(hours=2)
    ).isoformat()

    async def fake_mcp_tool(tool_name, arguments):
        assert tool_name == "check_measurement_quality"
        assert arguments["measured_at"] == actual_timestamp

        return {
            "tool_name": "measurement_quality",
            "is_fresh": True,
            "data_status": "fresh",
            "eligible_for_live_evaluation": True,
            "measured_at": incorrect_timestamp,
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
                "utc": actual_timestamp,
            }
        },
    }

    with pytest.raises(
        ValueError,
        match="timestamp mismatch",
    ):
        await investigation_module.mcp_investigation_node({
            "measurement": measurement,
        })
