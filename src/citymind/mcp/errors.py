"""
Custom exceptions for CITYMIND MCP operations.

These exception types allow callers to distinguish
connection failures from invalid MCP responses
without relying on error message text.
"""


class MCPError(Exception):
    """Base exception for MCP-related failures."""


class MCPConnectionError(MCPError):
    """MCP transport or connection failure."""


class MCPInvalidResponseError(MCPError):
    """MCP returned an invalid or inconsistent response."""


class MCPToolExecutionError(MCPError):
    """MCP server reported a tool execution failure."""
