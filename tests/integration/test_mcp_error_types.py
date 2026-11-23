"""
Tests for CITYMIND MCP exception types.
"""

import pytest

from citymind.mcp.errors import (
    MCPError,
    MCPConnectionError,
    MCPInvalidResponseError,
)


def test_mcp_error_inheritance():
    assert issubclass(MCPConnectionError, MCPError)
    assert issubclass(MCPInvalidResponseError, MCPError)

    assert issubclass(MCPError, Exception)


def test_mcp_connection_error_can_be_caught():
    with pytest.raises(
        MCPConnectionError,
        match="server unavailable",
    ):
        raise MCPConnectionError("server unavailable")


def test_mcp_invalid_response_error_can_be_caught():
    with pytest.raises(
        MCPInvalidResponseError,
        match="invalid quality response",
    ):
        raise MCPInvalidResponseError(
            "invalid quality response"
        )


def test_mcp_error_types_are_distinct():
    assert not issubclass(
        MCPConnectionError,
        MCPInvalidResponseError,
    )

    assert not issubclass(
        MCPInvalidResponseError,
        MCPConnectionError,
    )
