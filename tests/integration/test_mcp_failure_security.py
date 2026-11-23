"""
Verify that MCP failures do not produce
successful investigation results.
"""

from datetime import datetime, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_connection_failure_stops_investigation(
    monkeypatch,
):
    async def fake_mcp_tool(tool_name, arguments):
        raise ConnectionError("MCP server unavailable")

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
        ConnectionError,
        match="MCP server unavailable",
    ):
        await investigation_module.mcp_investigation_node({
            "measurement": measurement,
        })
