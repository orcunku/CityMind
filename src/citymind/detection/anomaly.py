from datetime import datetime

from citymind.ingestion.data_quality import (
    check_measurement_freshness,
)


# ============================================================
# CITYMIND - PM2.5 ANOMALY DETECTION
# ============================================================

# Prototype thresholds (not official health limits)
PM25_REVIEW_THRESHOLD = 15.0
PM25_HIGH_THRESHOLD = 35.0


def detect_pm25_anomaly(
    measurement: dict,
    now: datetime | None = None,
    quality: dict | None = None,
) -> dict:
    """
    Classify a PM2.5 measurement.

    If quality is provided, reuse its freshness
    result instead of calculating freshness again.

    If quality is not provided, calculate freshness
    using the original data quality function.

    Historical measurements cannot trigger
    live alerts.
    """

    # --------------------------------------------------------
    # STEP 1 - CHECK OR REUSE DATA QUALITY
    # --------------------------------------------------------

    if quality is None:
        quality = check_measurement_freshness(
            measurement,
            now=now,
        )

    # --------------------------------------------------------
    # STEP 2 - VALIDATE PM2.5 VALUE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # STEP 3 - CLASSIFY SEVERITY
    # --------------------------------------------------------

    if value > PM25_HIGH_THRESHOLD:
        severity = "high"

    elif value > PM25_REVIEW_THRESHOLD:
        severity = "review"

    else:
        severity = "normal"

    # --------------------------------------------------------
    # STEP 4 - INTERPRET QUALITY RESULT
    # --------------------------------------------------------

    is_fresh = quality["is_fresh"]

    # Supports the original quality checker schema
    # and the measurement quality tool schema.
    data_status = quality.get(
        "data_status",
        quality.get("status"),
    )

    if data_status not in ("fresh", "stale"):
        raise ValueError(
            "Invalid measurement quality status"
        )

    # Fail closed if quality fields disagree.
    if is_fresh is not True:
        data_status = "stale"

    if data_status != "fresh":
        is_fresh = False

    # --------------------------------------------------------
    # STEP 5 - RETURN ANOMALY ANALYSIS
    # --------------------------------------------------------

    return {
        "pollutant": "pm25",
        "value": value,
        "unit": "µg/m³",
        "severity": severity,
        "data_status": data_status,
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
