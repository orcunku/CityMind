from datetime import datetime, timedelta, timezone

from citymind.agents.graph import citymind_graph


def test_quality_result_is_saved_in_agent_state():
    """
    Verify that the measurement quality tool result
    is stored in the LangGraph agent state.
    """

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

    result = citymind_graph.invoke({
        "measurement": measurement
    })

    assert "quality_result" in result

    quality = result["quality_result"]

    assert quality["tool_name"] == "measurement_quality"
    assert quality["data_status"] == "fresh"
    assert quality["is_fresh"] is True
    assert quality["eligible_for_live_evaluation"] is True

    assert "measured_at" in quality
    assert "age_hours" in quality

    assert result["analysis"]["data_status"] == "fresh"
    assert result["report"]["live_alert_created"] is False
