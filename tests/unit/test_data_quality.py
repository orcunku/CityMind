from datetime import datetime, timedelta, timezone

import pytest

from citymind.ingestion.data_quality import (
    check_measurement_freshness,
    parse_utc_datetime,
)


FIXED_NOW = datetime(
    2026, 11, 23, 12, 0, tzinfo=timezone.utc
)


def make_measurement(measured_at):
    """Test için örnek OpenAQ ölçümü oluşturur."""
    return {
        "value": 35.0,
        "period": {
            "datetimeTo": {
                "utc": measured_at.isoformat()
            }
        },
    }


def test_parse_utc_datetime():
    result = parse_utc_datetime(
        "2022-10-25T14:00:00Z"
    )

    assert result.year == 2022
    assert result.hour == 14
    assert result.tzinfo == timezone.utc


def test_fresh_measurement():
    measured_at = FIXED_NOW - timedelta(hours=1)

    result = check_measurement_freshness(
        make_measurement(measured_at),
        now=FIXED_NOW,
    )

    assert result["status"] == "fresh"
    assert result["is_fresh"] is True
    assert result["eligible_for_live_alert"] is True


def test_stale_measurement():
    measured_at = FIXED_NOW - timedelta(days=30)

    result = check_measurement_freshness(
        make_measurement(measured_at),
        now=FIXED_NOW,
    )

    assert result["status"] == "stale"
    assert result["is_fresh"] is False
    assert result["eligible_for_live_alert"] is False


def test_future_measurement():
    measured_at = FIXED_NOW + timedelta(hours=1)

    result = check_measurement_freshness(
        make_measurement(measured_at),
        now=FIXED_NOW,
    )

    assert result["status"] == "stale"
    assert result["eligible_for_live_alert"] is False


def test_exactly_24_hours_old():
    measured_at = FIXED_NOW - timedelta(hours=24)

    result = check_measurement_freshness(
        make_measurement(measured_at),
        now=FIXED_NOW,
    )

    assert result["is_fresh"] is True


def test_more_than_24_hours_old():
    measured_at = FIXED_NOW - timedelta(
        hours=24, seconds=1
    )

    result = check_measurement_freshness(
        make_measurement(measured_at),
        now=FIXED_NOW,
    )

    assert result["is_fresh"] is False


def test_missing_timestamp():
    measurement = {
        "value": 35.0,
        "period": {},
    }

    with pytest.raises(
        ValueError,
        match="Measurement end timestamp is missing",
    ):
        check_measurement_freshness(
            measurement,
            now=FIXED_NOW,
        )


def test_timestamp_without_timezone():
    with pytest.raises(
        ValueError,
        match="Timestamp must include timezone",
    ):
        parse_utc_datetime(
            "2026-11-23T12:00:00"
        )