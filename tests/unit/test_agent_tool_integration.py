from datetime import datetime, timedelta, timezone

import citymind.agents.graph as graph_module


def test_agent_calls_measurement_quality_tool(monkeypatch):
    """
    Verify that the investigation node calls
    the measurement quality tool exactly once.
    """

    calls = []

    original_tool = graph_module.measurement_quality_tool

    def tracked_tool(measurement):
        calls.append(measurement)
        return original_tool(measurement)

    monkeypatch.setattr(
        graph_module,
        "measurement_quality_tool",
        tracked_tool,
    )

    measured_at = (
        datetime.now(timezone.utc)
        - timedelta(hours=2)
    )

    measurement = {
        "value": 40,
        "period": {
            "datetimeTo": {
                "utc": measured_at.isoformat()
            }
        },
    }

    result = graph_module.citymind_graph.invoke({
        "measurement": measurement
    })

    assert len(calls) == 1
    assert calls[0] == measurement

    assert result["analysis"]["data_status"] == "fresh"
    assert result["decision"] == "requires_investigation"

    assert result["report"]["live_alert_created"] is False
