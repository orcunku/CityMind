from datetime import datetime, timezone


MAX_DATA_AGE_HOURS = 24


def parse_utc_datetime(value: str) -> datetime:
    """OpenAQ zaman damgasını UTC olarak ayrıştırır."""
    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if parsed.tzinfo is None:
        raise ValueError(
            "Timestamp must include timezone information"
        )

    return parsed.astimezone(timezone.utc)


def check_measurement_freshness(
    measurement: dict,
    now: datetime | None = None,
) -> dict:
    """Ölçümün güncel olup olmadığını kontrol eder."""

    period = measurement.get("period") or {}
    datetime_to = period.get("datetimeTo") or {}
    timestamp = datetime_to.get("utc")

    if not timestamp:
        raise ValueError(
            "Measurement end timestamp is missing"
        )

    measured_at = parse_utc_datetime(timestamp)

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("Current time must be timezone-aware")

    now = now.astimezone(timezone.utc)

    age_hours = (
        now - measured_at
    ).total_seconds() / 3600

    is_fresh = 0 <= age_hours <= MAX_DATA_AGE_HOURS

    return {
        "measured_at": measured_at.isoformat(),
        "age_hours": round(age_hours, 2),
        "is_fresh": is_fresh,
        "eligible_for_live_alert": is_fresh,
        "status": "fresh" if is_fresh else "stale",
    }