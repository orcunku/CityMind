"""
CITYMIND MCP Client Bridge.

Calls tools through the CityMind MCP server
using the standard MCP protocol.
"""

import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from citymind.mcp.errors import MCPConnectionError


async def call_citymind_mcp_tool(
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    Start the local MCP server, call a tool,
    and return its result as a dictionary.

    Does not send live alerts.
    """

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [
            os.path.abspath("src"),
            existing_pythonpath,
        ]
        if part
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "citymind.mcp.server"],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (
            read,
            write,
        ):
            return await _call_tool_with_session(
                read, write, tool_name, arguments
            )
    except ConnectionError as exc:
        raise MCPConnectionError(
            "MCP connection failed"
        ) from exc


async def _call_tool_with_session(
    read, write, tool_name, arguments
):
    async with ClientSession(read, write) as session:
        await session.initialize()

        result = await session.call_tool(
            tool_name,
            arguments=arguments,
        )

        if result.isError:
            raise RuntimeError(
                f"MCP tool failed: {tool_name}"
            )

        if result.structuredContent is not None:
            data = result.structuredContent
        else:
            if not result.content:
                raise ValueError(
                    "MCP tool returned no content"
                )

            if result.content[0].type != "text":
                raise ValueError(
                    "Unexpected MCP response type"
                )

            data = json.loads(
                result.content[0].text
            )

        if not isinstance(data, dict):
            raise ValueError(
                "MCP tool must return a dictionary"
            )

        return data
