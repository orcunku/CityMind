from datetime import datetime, timedelta, timezone

import pytest

from citymind.agents.graph import citymind_graph


FIXED_NOW = datetime.now(timezone.utc)


def make_measurement(value, age_hours=1):
    measured_at = FIXED_NOW - timedelta(
        hours=age_hours
    )

    return {
        "value": value,
        "period": {
            "datetimeTo": {
                "utc": measured_at.isoformat()
            }
        },
    }


def run_agent(value, age_hours=1):
    return citymind_graph.invoke({
        "measurement": make_measurement(
            value,
            age_hours,
        )
    })


def test_historical_measurement():
    result = run_agent(
        34.19,
        age_hours=24 * 365,
    )

    assert (
        result["decision"]
        == "historical_analysis_only"
    )
    assert result["report"]["live_alert_created"] is False


def test_fresh_normal_measurement():
    result = run_agent(10)

    assert result["decision"] == "no_action_required"
    assert result["analysis"]["severity"] == "normal"


def test_fresh_review_measurement():
    result = run_agent(25)

    assert result["decision"] == "monitor_closely"
    assert result["analysis"]["severity"] == "review"


def test_fresh_high_measurement():
    result = run_agent(40)

    assert (
        result["decision"]
        == "requires_investigation"
    )
    assert result["analysis"]["severity"] == "high"
    assert result["report"]["live_alert_created"] is False


def test_future_measurement():
    result = run_agent(
        100,
        age_hours=-24,
    )

    assert (
        result["decision"]
        == "historical_analysis_only"
    )
    assert result["report"]["live_alert_created"] is False


def test_invalid_measurement():
    with pytest.raises(ValueError):
        citymind_graph.invoke({
            "measurement": {
                "value": -10,
                "period": {
                    "datetimeTo": {
                        "utc": FIXED_NOW.isoformat()
                    }
                },
            }
        })


def test_missing_measurement():
    with pytest.raises(ValueError):
        /-/- cxön cxöşpkouytfyıjkoşuydtrewfewrtyıuop-oıü Üind_graph.invoke({})-ü