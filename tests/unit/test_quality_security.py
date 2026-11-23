import pytest

from citymind.detection.anomaly import (
    detect_pm25_anomaly,
)


def make_measurement():
    return {
        "value": 40.0,
    }


def test_consistent_fresh_quality():
    """
    Fresh measurement with consistent
    quality information is accepted.
    """

    quality = {
        "is_fresh": True,
        "data_status": "fresh",
    }

    result = detect_pm25_anomaly(
        make_measurement(),
        quality=quality,
    )

    assert result["data_status"] == "fresh"
    assert result["severity"] == "high"
    assert result["eligible_for_live_alert"] is True


def test_consistent_stale_quality():
    """
    Stale measurement must not become
    eligible for a live alert.
    """

    quality = {
        "is_fresh": False,
        "data_status": "stale",
    }

    result = detect_pm25_anomaly(
        make_measurement(),
        quality=quality,
    )

    assert result["data_status"] == "stale"
    assert result["severity"] == "high"
    assert result["eligible_for_live_alert"] is False


def test_conflicting_quality_is_rejected():
    """
    Contradictory quality information
    must be rejected.
    """

    quality = {
        "is_fresh": True,
        "data_status": "stale",
    }

    with pytest.raises(
        ValueError,
        match="Conflicting freshness",
    ):
        detect_pm25_anomaly(
            make_measurement(),
            quality=quality,
        )
