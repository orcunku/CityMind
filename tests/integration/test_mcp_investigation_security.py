"""
Security tests for the MCP-backed investigation node.
"""

from datetime import datetime, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_investigation_rejects_conflicting_quality(
    monkeypatch,
):
    """
    Reject contradictory freshness information
    returned by the MCP client.
    """

    async def fake_mcp_tool(tool_name, arguments):
        assert tool_name == "check_measurement_quality"

        return {
            "tool_name": "measurement_quality",
            "is_fresh": True,
            "data_status": "stale",
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
                "utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        },
    }

    with pytest.raises(
        ValueError,
        match="Conflicting freshness",
    ):
        await investigation_module.mcp_investigation_node({
            "measurement": measurement,
        })
