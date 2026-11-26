"""Unit tests for secure structured LLM response parsing."""

import json

import pytest

from citymind.llm.errors import LLMResponseError
from citymind.llm.parser import parse_investigation_output
from citymind.llm.schemas import InvestigationOutput


@pytest.fixture
def valid_payload():
    return {
        "summary": "Additional monitoring is recommended.",
        "evidence": ["PM2.5 measurement is available."],
        "uncertainties": [],
        "recommendation": "continue_monitoring",
    }


def test_valid_json_response(valid_payload):
    result = parse_investigation_output(json.dumps(valid_payload))

    assert isinstance(result, InvestigationOutput)
    assert result.summary == valid_payload["summary"]
    assert result.evidence == tuple(valid_payload["evidence"])
    assert result.uncertainties == ()
    assert result.recommendation == "continue_monitoring"


def test_json_with_surrounding_whitespace(valid_payload):
    response = "\n  " + json.dumps(valid_payload) + "\n  "

    result = parse_investigation_output(response)

    assert result.summary == valid_payload["summary"]


@pytest.mark.parametrize(
    "response",
    [None, 123, True, {}, [], b"{}", "", "   "],
)
def test_invalid_response_input(response):
    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured response$",
    ):
        parse_investigation_output(response)


@pytest.mark.parametrize(
    "response",
    [
        "{",
        '{"summary":}',
        '{"summary": "test",}',
        "{'summary': 'test'}",
        "{} {}",
        "```json\n{}\n```",
        "Analysis: {}",
    ],
)
def test_malformed_json(response):
    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)


@pytest.mark.parametrize(
    "response",
    ["[]", "null", "true", "123", '"text"'],
)
def test_non_object_json(response):
    with pytest.raises(
        LLMResponseError,
        match="^Expected a JSON object$",
    ):
        parse_investigation_output(response)


@pytest.mark.parametrize(
    "field",
    ["summary", "evidence", "uncertainties", "recommendation"],
)
def test_missing_schema_field(valid_payload, field):
    del valid_payload[field]

    with pytest.raises(
        LLMResponseError,
        match="^Structured response failed schema validation$",
    ):
        parse_investigation_output(json.dumps(valid_payload))


def test_unauthorized_extra_field(valid_payload):
    valid_payload["execute_action"] = True

    with pytest.raises(
        LLMResponseError,
        match="^Structured response failed schema validation$",
    ):
        parse_investigation_output(json.dumps(valid_payload))


def test_unauthorized_recommendation(valid_payload):
    valid_payload["recommendation"] = "send_live_alert"

    with pytest.raises(
        LLMResponseError,
        match="^Structured response failed schema validation$",
    ):
        parse_investigation_output(json.dumps(valid_payload))

@pytest.mark.parametrize(
    "response",
    [
        '{"summary": "safe", "summary": "unsafe"}',
        '{"nested": {"key": 1, "key": 2}}',
        r'{"summary": "safe", "\u0073ummary": "unsafe"}',
    ],
)
def test_duplicate_json_keys(response):
    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)

@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity"],
)
def test_nonfinite_json_constants(constant):
    response = '{"value": ' + constant + '}'

    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)

def test_oversized_ascii_response():
    response = "x" * 65537

    with pytest.raises(
        LLMResponseError,
        match="^Structured response exceeds size limit$",
    ):
        parse_investigation_output(response)

def test_oversized_unicode_response():
    response = "ş" * 32769

    assert len(response) < 65536
    assert len(response.encode("utf-8")) > 65536

    with pytest.raises(
        LLMResponseError,
        match="^Structured response exceeds size limit$",
    ):
        parse_investigation_output(response)

def test_excessive_json_nesting():
    response = "[" * 10000 + "0" + "]" * 10000

    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)

def test_invalid_unicode_encoding():
    response = '{"summary": "\ud800"}'

    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)

def test_valid_unicode_response(valid_payload):
    valid_payload["summary"] = "İzmir hava kalitesi incelemesi."

    result = parse_investigation_output(
        json.dumps(valid_payload, ensure_ascii=False)
    )

    assert result.summary == valid_payload["summary"]

def test_exact_response_size_limit(valid_payload):
    response = json.dumps(valid_payload)
    response += " " * (65536 - len(response))

    assert len(response.encode("utf-8")) == 65536

    result = parse_investigation_output(response)

    assert result.summary == valid_payload["summary"]


@pytest.mark.parametrize(
    "field",
    ["summary", "evidence", "uncertainties"],
)
@pytest.mark.parametrize(
    "invalid_character",
    ["\ud800", "\udfff"],
)
def test_escaped_invalid_unicode(
    valid_payload, field, invalid_character
):
    if field == "summary":
        valid_payload[field] = invalid_character
    else:
        valid_payload[field] = [invalid_character]

    response = json.dumps(valid_payload, ensure_ascii=True)

    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)


def test_invalid_unicode_json_key(valid_payload):
    valid_payload["\ud800"] = "invalid key"

    response = json.dumps(valid_payload, ensure_ascii=True)

    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)


def test_valid_unicode_surrogate_pair(valid_payload):
    valid_payload["summary"] = "Environmental monitoring 🌍"

    response = json.dumps(valid_payload, ensure_ascii=True)

    result = parse_investigation_output(response)

    assert result.summary == valid_payload["summary"]


def test_escaped_turkish_characters(valid_payload):
    valid_payload["summary"] = "İzmir hava kalitesi ölçümü."

    response = json.dumps(valid_payload, ensure_ascii=True)

    result = parse_investigation_output(response)

    assert result.summary == valid_payload["summary"]


def test_oversized_whitespace_response():
    with pytest.raises(
        LLMResponseError,
        match="^Structured response exceeds size limit$",
    ):
        parse_investigation_output(" " * 65537)


@pytest.mark.parametrize(
    "number",
    ["1e999", "-1e999"],
)
def test_json_numeric_overflow(number):
    response = '{"value": ' + number + '}'

    with pytest.raises(
        LLMResponseError,
        match="^Invalid structured JSON response$",
    ):
        parse_investigation_output(response)
