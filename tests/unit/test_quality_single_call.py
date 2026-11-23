from datetime import datetime, timedelta, timezone

import citymind.ingestion.data_quality as data_quality
import citymind.tools.measurement_quality as quality_module


def test_quality_checked_once_in_agent(monkeypatch):
    """
    Verify that freshness is calculated only once
    during a LangGraph execution.
    """

    from citymind.agents.graph import citymind_graph

    calls = []

    original_checker = (
        quality_module.check_measurement_freshness
    )

    def tracked_checker(measurement, now=None):
        calls.append(measurement)
        return original_checker(
            measurement,
            now=now,
        )

    monkeypatch.setattr(
        quality_module,
        "check_measurement_freshness",
        tracked_checker,
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

    result = citymind_graph.invoke({
        "measurement": measurement
    })

    assert len(calls) == 1

    assert result["quality_result"]["is_fresh"] is True

    assert (
        result["analysis"]["eligible_for_live_alert"]
        is True
    )

    assert result["report"]["live_alert_created"] is False
