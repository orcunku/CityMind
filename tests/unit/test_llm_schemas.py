"""Unit tests for structured LLM output schemas."""

import pytest
from pydantic import ValidationError

from citymind.llm.schemas import InvestigationOutput


@pytest.fixture
def valid_output():
    return {
        "summary": "Additional monitoring is recommended.",
        "evidence": ["PM2.5 measurement is available."],
        "uncertainties": ["Data freshness requires review."],
        "recommendation": "continue_monitoring",
    }


def test_valid_output(valid_output):
    result = InvestigationOutput.model_validate(valid_output)

    assert result.summary == valid_output["summary"]
    assert result.evidence == valid_output["evidence"]
    assert result.uncertainties == valid_output["uncertainties"]
    assert result.recommendation == "continue_monitoring"


@pytest.mark.parametrize(
    "recommendation",
    [
        "no_additional_review",
        "continue_monitoring",
        "request_additional_data",
        "request_human_review",
    ],
)
def test_allowed_recommendations(valid_output, recommendation):
    valid_output["recommendation"] = recommendation

    result = InvestigationOutput.model_validate(valid_output)

    assert result.recommendation == recommendation


@pytest.mark.parametrize(
    "field",
    ["summary", "evidence", "uncertainties", "recommendation"],
)
def test_missing_required_field(valid_output, field):
    del valid_output[field]

    with pytest.raises(ValidationError):
        InvestigationOutput.model_validate(valid_output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("live_alert_created", True),
        ("execute_action", True),
        ("severity", "high"),
        ("decision", "send_alert"),
    ],
)
def test_forbidden_extra_fields(valid_output, field, value):
    valid_output[field] = value

    with pytest.raises(ValidationError):
        InvestigationOutput.model_validate(valid_output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", 123),
        ("summary", None),
        ("evidence", "Not a list"),
        ("evidence", [123]),
        ("uncertainties", None),
        ("recommendation", True),
    ],
)
def test_invalid_field_types(valid_output, field, value):
    valid_output[field] = value

    with pytest.raises(ValidationError):
        InvestigationOutput.model_validate(valid_output)


def test_invalid_recommendation(valid_output):
    valid_output["recommendation"] = "send_live_alert"

    with pytest.raises(ValidationError):
        InvestigationOutput.model_validate(valid_output)
