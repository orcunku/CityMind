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
    assert result.evidence == tuple(valid_output["evidence"])
    assert result.uncertainties == tuple(valid_output["uncertainties"])
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

@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", ""),
        ("summary", "   "),
        ("summary", "x" * 2001),
        ("evidence", []),
        ("evidence", [""]),
        ("evidence", ["   "]),
        ("evidence", ["x" * 501]),
        ("evidence", ["Evidence"] * 11),
        ("uncertainties", [""]),
        ("uncertainties", ["   "]),
        ("uncertainties", ["x" * 501]),
        ("uncertainties", ["Uncertainty"] * 11),
    ],
)
def test_invalid_field_boundaries(valid_output, field, value):
    valid_output[field] = value

    with pytest.raises(ValidationError):
        InvestigationOutput.model_validate(valid_output)

def test_maximum_allowed_boundaries(valid_output):
    valid_output["summary"] = "x" * 2000
    valid_output["evidence"] = ["x" * 500] * 10
    valid_output["uncertainties"] = ["x" * 500] * 10

    result = InvestigationOutput.model_validate(valid_output)

    assert len(result.summary) == 2000
    assert len(result.evidence) == 10
    assert len(result.uncertainties) == 10

def test_empty_uncertainties_allowed(valid_output):
    valid_output["uncertainties"] = []

    result = InvestigationOutput.model_validate(valid_output)

    assert result.uncertainties == ()

def test_whitespace_normalization(valid_output):
    valid_output["summary"] = "  Analysis summary  "
    valid_output["evidence"] = ["  Evidence  "]
    valid_output["uncertainties"] = ["  Uncertainty  "]

    result = InvestigationOutput.model_validate(valid_output)

    assert result.summary == "Analysis summary"
    assert result.evidence == ("Evidence",)
    assert result.uncertainties == ("Uncertainty",)

@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", "Modified summary"),
        ("evidence", ["Modified evidence"]),
        ("uncertainties", []),
        ("recommendation", "request_human_review"),
    ],
)
def test_frozen_model_fields(valid_output, field, value):
    result = InvestigationOutput.model_validate(valid_output)

    with pytest.raises(ValidationError) as exc:
        setattr(result, field, value)

    assert exc.value.errors()[0]["type"] == "frozen_instance"


@pytest.mark.parametrize("field", ["evidence", "uncertainties"])
def test_sequence_is_immutable(valid_output, field):
    result = InvestigationOutput.model_validate(valid_output)
    sequence = getattr(result, field)

    assert isinstance(sequence, tuple)

    with pytest.raises(AttributeError):
        sequence.append("Unauthorized content")


@pytest.mark.parametrize("field", ["evidence", "uncertainties"])
def test_sequence_item_cannot_be_modified(valid_output, field):
    result = InvestigationOutput.model_validate(valid_output)
    sequence = getattr(result, field)

    with pytest.raises(TypeError):
        sequence[0] = "Unauthorized content"


@pytest.mark.parametrize("field", ["evidence", "uncertainties"])
def test_original_input_mutation_is_isolated(valid_output, field):
    result = InvestigationOutput.model_validate(valid_output)
    original = getattr(result, field)

    valid_output[field].append("Unauthorized content")

    assert getattr(result, field) == original
    assert len(getattr(result, field)) == 1


def test_json_round_trip(valid_output):
    import json

    result = InvestigationOutput.model_validate_json(
        json.dumps(valid_output)
    )

    assert isinstance(result.evidence, tuple)
    assert isinstance(result.uncertainties, tuple)
    assert json.loads(result.model_dump_json()) == valid_output

    restored = InvestigationOutput.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result
