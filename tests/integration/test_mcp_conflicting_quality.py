"""
Security test: contradictory MCP quality data
must raise MCPInvalidResponseError.
"""

from datetime import datetime, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module
from citymind.mcp.errors import MCPInvalidResponseError


@pytest.mark.anyio
async def test_mcp_rejects_conflicting_quality(monkeypatch):
    measured_at = datetime.now(timezone.utc).isoformat()

    async def fake_mcp_tool(tool_name, arguments):
        assert tool_name == "check_measurement_quality"

        return {
            "tool_name": "measurement_quality",
            "is_fresh": True,
            "data_status": "stale",
            "eligible_for_live_evaluation": True,
            "measured_at": measured_at,
            "age_hours": 0.0,
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
                "utc": measured_at,
            }
        },
    }

    with pytest.raises(
        MCPInvalidResponseError,
        match="Invalid MCP quality data",
    ):
        await investigation_module.mcp_investigation_node({
            "measurement": measurement,
        })
