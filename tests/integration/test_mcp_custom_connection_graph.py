"""
Verify that LangGraph safely handles MCPConnectionError.
"""

from unittest.mock import patch

import pytest

from citymind.agents.mcp_graph import citymind_mcp_graph
from citymind.mcp.errors import MCPConnectionError


@pytest.mark.anyio
async def test_graph_handles_custom_mcp_connection_error():

    async def failing_mcp_investigation(state):
        raise MCPConnectionError("MCP server unavailable")

    measurement = {
        "value": 40.0,
        "period": {
            "datetimeTo": {
                "utc": "2022-10-25T12:00:00Z"
            }
        },
    }

    with patch(
        "citymind.agents.mcp_graph.mcp_investigation_node",
        failing_mcp_investigation,
    ):
        result = await citymind_mcp_graph.ainvoke(
            {"measurement": measurement}
        )

    assert result["decision"] == "investigation_unavailable"
    assert result["report"]["data_status"] == "unavailable"
    assert result["report"]["error_type"] == "mcp_connection_error"
    assert result["report"]["live_alert_created"] is False
