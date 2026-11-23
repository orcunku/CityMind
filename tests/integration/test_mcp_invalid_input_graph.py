"""
Security test: invalid measurement input must
not be disguised as an MCP response failure.
"""

from datetime import datetime, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module
from citymind.agents.mcp_graph import citymind_mcp_graph


@pytest.mark.anyio
async def test_invalid_measurement_is_not_mcp_failure(
    monkeypatch,
):
    measured_at = datetime.now(timezone.utc).isoformat()

    async def fake_mcp_tool(tool_name, arguments):
        assert tool_name == "check_measurement_quality"

        return {
            "tool_name": "measurement_quality",
            "is_fresh": True,
            "data_status": "fresh",
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
        "value": "invalid",
        "period": {
            "datetimeTo": {
                "utc": measured_at,
            }
        },
    }

    with pytest.raises(
        ValueError,
        match="PM2.5 measurement must be numeric",
    ):
        await citymind_mcp_graph.ainvoke({
            "measurement": measurement,
        })
