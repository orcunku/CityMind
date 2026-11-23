from datetime import datetime, timedelta, timezone

import pytest

from citymind.agents.graph import citymind_graph


# ============================================================
# CITYMIND - LANGGRAPH AGENT TESTS
# ============================================================

FIXED_NOW = datetime.now(timezone.utc)


def make_measurement(value, age_hours=1):
    """
    Test için örnek PM2.5 ölçümü oluşturur.

    age_hours:
        1   = 1 saat önce
        100 = 100 saat önce
        -24 = 24 saat sonra (gelecek tarih)
    """

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
    """
    CityMind LangGraph ajanını çalıştırır.
    """

    measurement = make_measurement(
        value,
        age_hours=age_hours,
    )

    return citymind_graph.invoke({
        "measurement": measurement
    })


# ============================================================
# TEST 1 - HISTORICAL MEASUREMENT
# ============================================================

def test_historical_measurement():
    result = run_agent(
        34.19,
        age_hours=24 * 365,
    )

    assert (
        result["decision"]
        == "historical_analysis_only"
    )

    assert (
        result["report"]["live_alert_created"]
        is False
    )


# ============================================================
# TEST 2 - FRESH NORMAL MEASUREMENT
# ============================================================

def test_fresh_normal_measurement():
    result = run_agent(10)

    assert (
        result["decision"]
        == "no_action_required"
    )

    assert (
        result["analysis"]["severity"]
        == "normal"
    )


# ============================================================
# TEST 3 - FRESH REVIEW MEASUREMENT
# ============================================================

def test_fresh_review_measurement():
    result = run_agent(25)

    assert (
        result["decision"]
        == "monitor_closely"
    )

    assert (
        result["analysis"]["severity"]
        == "review"
    )


# ============================================================
# TEST 4 - FRESH HIGH MEASUREMENT
# ============================================================

def test_fresh_high_measurement():
    result = run_agent(40)

    assert (
        result["decision"]
        == "requires_investigation"
    )

    assert (
        result["analysis"]["severity"]
        == "high"
    )

    assert (
        result["report"]["live_alert_created"]
        is False
    )


# ============================================================
# TEST 5 - FUTURE MEASUREMENT
# ============================================================

def test_future_measurement():
    result = run_agent(
        100,
        age_hours=-24,
    )

    assert (
        result["decision"]
        == "historical_analysis_only"
    )

    assert (
        result["report"]["live_alert_created"]
        is False
    )


# ============================================================
# TEST 6 - INVALID MEASUREMENT
# ============================================================

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


# ============================================================
# TEST 7 - MISSING MEASUREMENT
# ============================================================

def test_missing_measurement():
    with pytest.raises(ValueError):
        citymind_graph.invoke({})


# ============================================================
# TEST 8 - HISTORICAL ROUTING
# ============================================================

def test_historical_routing():
    """
    Eski verinin historical node'a
    yönlendirildiğini doğrular.
    """

    measurement = make_measurement(
        40,
        age_hours=100,
    )

    visited = []

    for event in citymind_graph.stream({
        "measurement": measurement
    }):
        visited.extend(event.keys())

    assert visited == [
        "monitoring",
        "investigation",
        "historical",
        "action",
    ]


# ============================================================
# TEST 9 - FRESH ROUTING
# ============================================================

def test_fresh_routing():
    """
    Güncel verinin decision node'a
    yönlendirildiğini doğrular.
    """

    measurement = make_measurement(
        40,
        age_hours=1,
    )

    visited = []

    for event in citymind_graph.stream({
        "measurement": measurement
    }):
        visited.extend(event.keys())

    assert visited == [
        "monitoring",
        "investigation",
        "decision",
        "action",
    ]