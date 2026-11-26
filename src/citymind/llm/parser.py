"""Secure JSON parsing for structured LLM responses."""

import json

from pydantic import ValidationError

from citymind.llm.errors import LLMResponseError
from citymind.llm.schemas import InvestigationOutput


MAX_RESPONSE_BYTES = 65536


def _reject_duplicate_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")

        result[key] = value

    return result


def _reject_nonfinite_constant(value):
    raise ValueError("Invalid JSON numeric constant")


def parse_investigation_output(
    response: str,
) -> InvestigationOutput:
    """Validate untrusted LLM output without executing it."""

    if not isinstance(response, str):
        raise LLMResponseError("Invalid structured response")

    if len(response) > MAX_RESPONSE_BYTES:
        raise LLMResponseError("Structured response exceeds size limit")

    if not response.strip():
        raise LLMResponseError("Invalid structured response")

    try:
        if len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise LLMResponseError(
                "Structured response exceeds size limit"
            )

        payload = json.loads(
            response,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )

    except (ValueError, RecursionError):
        raise LLMResponseError(
            "Invalid structured JSON response"
        ) from None

    if not isinstance(payload, dict):
        raise LLMResponseError("Expected a JSON object")

    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError):
        raise LLMResponseError(
            "Invalid structured JSON response"
        ) from None

    try:
        return InvestigationOutput.model_validate(payload)

    except ValidationError:
        raise LLMResponseError(
            "Structured response failed schema validation"
        ) from None
