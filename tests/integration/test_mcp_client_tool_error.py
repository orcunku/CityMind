"""
Verify MCP tool execution errors are classified correctly.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from citymind.mcp.client import _call_tool_with_session
from citymind.mcp.errors import MCPToolExecutionError


@pytest.mark.anyio
async def test_mcp_client_classifies_tool_execution_error():

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, tool_name, arguments):
            return SimpleNamespace(
                isError=True,
                structuredContent=None,
                content=[],
            )

    with patch(
        "citymind.mcp.client.ClientSession",
        return_value=FakeSession(),
    ):
        with pytest.raises(
            MCPToolExecutionError,
            match="MCP tool failed",
        ):
            await _call_tool_with_session(
                None,
                None,
                "check_measurement_quality",
                {"value": 40.0},
            )
