"""
Security test for MCP client connection failures.
"""

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from citymind.mcp.client import call_citymind_mcp_tool
from citymind.mcp.errors import MCPConnectionError


@pytest.mark.anyio
async def test_mcp_client_raises_connection_error():

    @asynccontextmanager
    async def failing_stdio_client(*args, **kwargs):
        raise ConnectionError("MCP server unavailable")
        yield

    with patch(
        "citymind.mcp.client.stdio_client",
        failing_stdio_client,
    ):
        with pytest.raises(
            MCPConnectionError,
            match="MCP",
        ):
            await call_citymind_mcp_tool(
                "check_measurement_quality",
                {
                    "value": 40.0,
                    "measured_at": "2026-01-01T00:00:00Z",
                },
            )
