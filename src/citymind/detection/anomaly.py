from datetime import datetime

from citymind.ingestion.data_quality import (
    check_measurement_freshness,
)


# ============================================================
# CITYMIND - PM2.5 ANOMALY DETECTION
# ============================================================

# Prototype thresholds - not official health limits
PM25_REVIEW_THRESHOLD = 15.0
PM25_HIGH_THRESHOLD = 35.0


# ============================================================
# QUALITY VALIDATION
# ============================================================

def validate_quality_result(quality: dict) -> dict:
    """
    Validate the structure and consistency
    of a measurement quality result.

    Reject contradictory or malformed results.
    """

    if not isinstance(quality, dict):
        raise ValueError(
            "Measurement quality must be a dictionary"
        )

    if "is_fresh" not in quality:
        raise ValueError(
            "Measurement quality is missing is_fresh"
        )

    is_fresh = quality["is_fresh"]

    if not isinstance(is_fresh, bool):
        raise ValueError(
            "Measurement quality is_fresh must be boolean"
        )

    # Supports the original checker schema ("status")
    # and the measurement tool schema ("data_status").
    has_data_status = "data_status" in quality
    has_status = "status" in quality

    if not has_data_status and not has_status:
        raise ValueError(
            "Measurement quality status is missing"
        )

    if has_data_status and has_status:
        if quality["data_status"] != quality["status"]:
            raise ValueError(
                "Conflicting measurement quality statuses"
            )

    data_status = (
        quality["data_status"]
        if has_data_status
        else quality["status"]
    )

    if data_status not in ("fresh", "stale"):
        raise ValueError(
            "Invalid measurement quality status"
        )

    if is_fresh != (data_status == "fresh"):
        raise ValueError(
            "Conflicting freshness and status values"
        )

    return {
        "is_fresh": is_fresh,
        "data_status": data_status,
    }


# ============================================================
# PM2.5 ANOMALY DETECTION
# ============================================================

def detect_pm25_anomaly(
    measurement: dict,
    now: datetime | None = None,
    quality: dict | None = None,
) -> dict:
    """
    Classify a PM2.5 measurement.

    If quality is provided, reuse it instead
    of calculating freshness again.

    Invalid quality results are rejected.

    Historical measurements cannot trigger
    live alerts.
    """

    # --------------------------------------------------------
    # STEP 1 - GET QUALITY RESULT
    # --------------------------------------------------------

    if quality is None:
        quality = check_measurement_freshness(
            measurement,
            now=now,
        )

    validated_quality = validate_quality_result(
        quality
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
    # STEP 4 - APPLY FRESHNESS SAFETY RULES
    # --------------------------------------------------------

    is_fresh = validated_quality["is_fresh"]
    data_status = validated_quality["data_status"]

    eligible_for_live_alert = (
        is_fresh and severity == "high"
    )

    # --------------------------------------------------------
    # STEP 5 - RETURN ANALYSIS
    # --------------------------------------------------------

    return {
        "pollutant": "pm25",
        "value": value,
        "unit": "µg/m³",
        "severity": severity,
        "data_status": data_status,
        "is_anomaly": severity != "normal",
        "eligible_for_live_alert": (
            eligible_for_live_alert
        ),
        "reason": (
            "Historical measurement: live alert blocked"
            if not is_fresh
            else f"PM2.5 classified as {severity}"
        ),
    }
