"""
CITYMIND runtime agent contract validation tests.
"""

import pytest

from citymind.agents.contracts import (
    AGENT_CONTRACTS,
    AgentContractError,
    get_agent_contract,
    validate_agent_inputs,
    validate_agent_outputs,
)


def test_get_registered_contract():
    assert get_agent_contract("investigation") == (
        AGENT_CONTRACTS["investigation"]
    )


def test_unknown_agent_is_rejected():
    with pytest.raises(
        AgentContractError,
        match="Unknown agent",
    ):
        get_agent_contract("unknown_agent")


def test_coordinator_accepts_empty_state():
    validate_agent_inputs("coordinator", {})


def test_valid_investigation_input():
    validate_agent_inputs(
        "investigation",
        {"measurement": {"value": 25.0}},
    )


def test_missing_required_input():
    with pytest.raises(
        AgentContractError,
        match="missing input fields: measurement",
    ):
        validate_agent_inputs("investigation", {})


def test_missing_multiple_inputs():
    with pytest.raises(
        AgentContractError,
        match="missing input fields: analysis, decision",
    ):
        validate_agent_inputs("action", {})


@pytest.mark.parametrize(
    "state",
    [None, [], "invalid", 42],
)
def test_invalid_input_type(state):
    with pytest.raises(
        AgentContractError,
        match="input state must be a dictionary",
    ):
        validate_agent_inputs("coordinator", state)


def test_valid_investigation_output():
    validate_agent_outputs(
        "investigation",
        {
            "quality_result": {},
            "analysis": {},
        },
    )


def test_missing_required_output():
    with pytest.raises(
        AgentContractError,
        match="missing output fields: analysis",
    ):
        validate_agent_outputs(
            "investigation",
            {"quality_result": {}},
        )


def test_missing_multiple_outputs():
    with pytest.raises(
        AgentContractError,
        match="missing output fields: quality_result, analysis",
    ):
        validate_agent_outputs("investigation", {})


@pytest.mark.parametrize(
    "result",
    [None, [], "invalid", 42],
)
def test_invalid_output_type(result):
    with pytest.raises(
        AgentContractError,
        match="output must be a dictionary",
    ):
        validate_agent_outputs("action", result)


def test_additional_input_fields_are_allowed():
    validate_agent_inputs(
        "investigation",
        {
            "measurement": {},
            "monitoring_status": "measurement_received",
        },
    )


def test_input_validation_does_not_mutate_state():
    state = {"measurement": {"value": 25.0}}

    validate_agent_inputs("investigation", state)

    assert state == {
        "measurement": {"value": 25.0},
    }


def test_output_validation_does_not_mutate_result():
    result = {"report": {"live_alert_created": False}}

    validate_agent_outputs("action", result)

    assert result == {
        "report": {"live_alert_created": False},
    }
