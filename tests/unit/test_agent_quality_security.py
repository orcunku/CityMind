from datetime import datetime, timezone

import pytest

import citymind.agents.graph as graph_module


def test_agent_rejects_conflicting_quality(monkeypatch):
    """
    The LangGraph workflow must reject
    contradictory quality information.
    """

    def fake_quality_tool(measurement):
        return {
            "tool_name": "measurement_quality",
            "is_fresh": True,
            "data_status": "stale",
        }

    monkeypatch.setattr(
        graph_module,
        "measurement_quality_tool",
        fake_quality_tool,
    )

    measurement = {
        "value": 40.0,
        "period": {
            "datetimeTo": {
                "utc": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        },
    }

    with pytest.raises(
        ValueError,
        match="Conflicting freshness",
    ):
        graph_module.citymind_graph.invoke(
            {"measurement": measurement}
        )
