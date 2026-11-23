"""
End-to-end tests for CITYMIND's
MCP-backed LangGraph workflow.
"""

from datetime import datetime, timedelta, timezone

import pytest

from citymind.agents.mcp_graph import (
    citymind_mcp_graph,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_measurement(age_hours: int) -> dict:
    measured_at = (
        datetime.now(timezone.utc)
        - timedelta(hours=age_hours)
    ).isoformat()

    return {
        "value": 40.0,
        "period": {
            "datetimeTo": {
                "utc": measured_at,
            }
        },
    }


@pytest.mark.anyio
async def test_mcp_graph_fresh_measurement():
    result = await citymind_mcp_graph.ainvoke({
        "measurement": make_measurement(2),
    })

    assert result["monitoring_status"] == "measurement_received"

    assert result["quality_result"]["is_fresh"] is True
    assert result["analysis"]["severity"] == "high"
    assert result["analysis"]["data_status"] == "fresh"

    assert result["decision"] == "requires_investigation"
    assert result["report"]["live_alert_created"] is False


@pytest.mark.anyio
async def test_mcp_graph_historical_measurement():
    result = await citymind_mcp_graph.ainvoke({
        "measurement": make_measurement(48),
    })

    assert result["monitoring_status"] == "measurement_received"

    assert result["quality_result"]["is_fresh"] is False
    assert result["analysis"]["data_status"] != "fresh"
    assert result["analysis"]["eligible_for_live_alert"] is False

    assert result["decision"] == "historical_analysis_only"
    assert result["report"]["live_alert_created"] is False
