from datetime import datetime, timedelta, timezone

import pytest

from citymind.detection.anomaly import (
    detect_pm25_anomaly,
)


FIXED_NOW = datetime(
    2026, 11, 23, 12, 0, tzinfo=timezone.utc
)


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


def test_normal_measurement():
    result = detect_pm25_anomaly(
        make_measurement(10),
        now=FIXED_NOW,
    )

    assert result["severity"] == "normal"
    assert result["is_anomaly"] is False
    assert result["eligible_for_live_alert"] is False


def test_review_measurement():
    result = detect_pm25_anomaly(
        make_measurement(25),
        now=FIXED_NOW,
    )

    assert result["severity"] == "review"
    assert result["is_anomaly"] is True
    assert result["eligible_for_live_alert"] is False


def test_high_measurement():
    result = detect_pm25_anomaly(
        make_measurement(40),
        now=FIXED_NOW,
    )

    assert result["severity"] == "high"
    assert result["is_anomaly"] is True
    assert result["eligible_for_live_alert"] is True


def test_stale_high_measurement():
    result = detect_pm25_anomaly(
        make_measurement(100, age_hours=48),
        now=FIXED_NOW,
    )

    assert result["severity"] == "high"
    assert result["data_status"] == "stale"
    assert result["eligible_for_live_alert"] is False


def test_review_threshold_boundary():
    result = detect_pm25_anomaly(
        make_measurement(15),
        now=FIXED_NOW,
    )

    assert result["severity"] == "normal"


def test_high_threshold_boundary():
    result = detect_pm25_anomaly(
        make_measurement(35),
        now=FIXED_NOW,
    )

    assert result["severity"] == "review"


def test_negative_value():
    with pytest.raises(ValueError):
        detect_pm25_anomaly(
            make_measurement(-5),
            now=FIXED_NOW,
        )


def test_non_numeric_value():
    with pytest.raises(ValueError):
        detect_pm25_anomaly(
            make_measurement("high"),
            now=FIXED_NOW,
        )


def test_missing_value():
    with pytest.raises(ValueError):
        detect_pm25_anomaly(
            make_measurement(None),
            now=FIXED_NOW,
        )


def test_infinite_value():
    with pytest.raises(ValueError):
        detect_pm25_anomaly(
            make_measurement(float("inf")),
            now=FIXED_NOW,
        )


def test_nan_value():
    with pytest.raises(ValueError):
        detect_pm25_anomaly(
            make_measurement(float("nan")),
            now=FIXED_NOW,
        )


def test_future_high_measurement():
    result = detect_pm25_anomaly(
        make_measurement(100, age_hours=-1),
        now=FIXED_NOW,
    )

    assert result["severity"] == "high"
    assert result["eligible_for_live_alert"] is False