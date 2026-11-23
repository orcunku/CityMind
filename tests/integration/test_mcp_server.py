import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_pm25_tool():
    """
    Start the CityMind MCP server,
    discover its tools and call analyze_pm25.
    """

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "citymind.mcp.server"],
        env=env,
    )

    async with stdio_client(server_params) as (
        read,
        write,
    ):
        async with ClientSession(
            read,
            write,
        ) as session:

            await session.initialize()

            tools = await session.list_tools()

            tool_names = [
                tool.name
                for tool in tools.tools
            ]

            assert "analyze_pm25" in tool_names

            measured_at = (
                datetime.now(timezone.utc)
                - timedelta(hours=2)
            ).isoformat()

            result = await session.call_tool(
                "analyze_pm25",
                arguments={
                    "value": 40.0,
                    "measured_at": measured_at,
                },
            )

            assert result.isError is False

            if result.structuredContent is not None:
                analysis = result.structuredContent
            else:
                assert result.content
                assert result.content[0].type == "text"

                analysis = json.loads(
                    result.content[0].text
                )

            assert analysis["severity"] == "high"

            assert analysis["data_status"] == "fresh"

            assert (
                analysis["eligible_for_live_alert"]
                is True
            )
