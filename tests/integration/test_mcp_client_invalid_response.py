"""
Verify that malformed MCP responses use
MCPInvalidResponseError.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from citymind.mcp.client import _call_tool_with_session
from citymind.mcp.errors import MCPInvalidResponseError


@pytest.mark.anyio
async def test_mcp_client_rejects_empty_response():

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, tool_name, arguments):
            return SimpleNamespace(
                isError=False,
                structuredContent=None,
                content=[],
            )

    with patch(
        "citymind.mcp.client.ClientSession",
        return_value=FakeSession(),
    ):
        with pytest.raises(
            MCPInvalidResponseError,
            match="MCP",
        ):
            await _call_tool_with_session(
                None,
                None,
                "check_measurement_quality",
                {
                    "value": 40.0,
                    "measured_at": "2026-01-01T00:00:00Z",
                },
            )
