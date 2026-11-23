from datetime import datetime

from citymind.ingestion.data_quality import (
    check_measurement_freshness,
)


PM25_REVIEW_THRESHOLD = 15.0
PM25_HIGH_THRESHOLD = 35.0


def detect_pm25_anomaly(
    measurement: dict,
    now: datetime | None = None,
) -> dict:
    """
    PM2.5 ölçümünü sınıflandırır.

    Eski ölçümler canlı alarm oluşturamaz.
    """

    quality = check_measurement_freshness(
        measurement,
        now=now,
    )

    value = measurement.get("value")

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            "PM2.5 measurement must be numeric"
        )

    if not 0 <= value < float("inf"):
        raise ValueError(
            "PM2.5 measurement must be finite and non-negative"
        )

    if value > PM25_HIGH_THRESHOLD:
        severity = "high"
    elif value > PM25_REVIEW_THRESHOLD:
        severity = "review"
    else:
        severity = "normal"

    is_fresh = quality["is_fresh"]

    return {
        "pollutant": "pm25",
        "value": value,
        "unit": "µg/m³",
        "severity": severity,
        "data_status": quality["status"],
        "is_anomaly": severity != "normal",
        "eligible_for_live_alert": (
            is_fresh and severity == "high"
        ),
        "reason": (
            "Historical measurement: live alert blocked"
            if not is_fresh
            else f"PM2.5 classified as {severity}"
        ),
    }