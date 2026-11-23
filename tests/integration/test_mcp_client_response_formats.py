"""
Security tests for malformed MCP response formats.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from citymind.mcp.client import _call_tool_with_session
from citymind.mcp.errors import MCPInvalidResponseError


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            isError=False,
            structuredContent=None,
            content=[
                SimpleNamespace(type="text", text="{invalid json")
            ],
        ),
        SimpleNamespace(
            isError=False,
            structuredContent=None,
            content=[
                SimpleNamespace(type="image")
            ],
        ),
        SimpleNamespace(
            isError=False,
            structuredContent=None,
            content=[
                SimpleNamespace(type="text", text="[1, 2, 3]")
            ],
        ),
        SimpleNamespace(
            isError=False,
            structuredContent=["invalid"],
            content=[],
        ),
    ],
    ids=[
        "malformed-json",
        "unexpected-content-type",
        "json-list",
        "invalid-structured-content",
    ],
)
async def test_invalid_mcp_response_formats(response):

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, tool_name, arguments):
            return response

    with patch(
        "citymind.mcp.client.ClientSession",
        return_value=FakeSession(),
    ):
        with pytest.raises(MCPInvalidResponseError):
            await _call_tool_with_session(
                None,
                None,
                "check_measurement_quality",
                {"value": 40.0},
            )
