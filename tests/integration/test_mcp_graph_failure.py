"""
Test safe LangGraph behavior when MCP is unavailable.
"""

from datetime import datetime, timezone

import pytest

import citymind.agents.mcp_investigation as investigation_module
from citymind.agents.mcp_graph import citymind_mcp_graph


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_graph_returns_safe_failure_report(
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

    result = await citymind_mcp_graph.ainvoke({
        "measurement": measurement,
    })

    assert result["decision"] == "investigation_unavailable"
    assert result["quality_result"]["data_status"] == "unavailable"
    assert result["report"]["live_alert_created"] is False
    assert result["report"]["error_type"] == "mcp_connection_error"
