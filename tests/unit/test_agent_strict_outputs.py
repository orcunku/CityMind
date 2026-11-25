"""
CITYMIND strict agent output validation tests.
"""

import pytest

from citymind.agents.contracts import (
    AGENT_CONTRACTS,
    AgentContractError,
    validate_agent_outputs,
)


@pytest.mark.parametrize(
    "agent_name",
    tuple(AGENT_CONTRACTS),
)
def test_all_agents_reject_unknown_output(agent_name):
    contract = AGENT_CONTRACTS[agent_name]

    result = {
        field: None
        for field in contract.expected_outputs
    }

    result["unauthorized_field"] = True

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields",
    ):
        validate_agent_outputs(agent_name, result)


def test_coordinator_cannot_override_decision():
    result = {
        "coordination_status": "assigned",
        "assigned_task": "investigate_measurement",
        "decision": "no_action_required",
    }

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields: decision",
    ):
        validate_agent_outputs("coordinator", result)


def test_investigation_cannot_create_report():
    result = {
        "quality_result": {},
        "analysis": {},
        "report": {"live_alert_created": True},
    }

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields: report",
    ):
        validate_agent_outputs("investigation", result)


def test_memory_cannot_override_decision():
    result = {
        "memory_status": "saved",
        "incident_id": 1,
        "vector_status": "indexed",
        "decision": "no_action_required",
    }

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields: decision",
    ):
        validate_agent_outputs("memory", result)


def test_action_cannot_override_analysis():
    result = {
        "report": {},
        "analysis": {"severity": "normal"},
    }

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields: analysis",
    ):
        validate_agent_outputs("action", result)


def test_multiple_unexpected_outputs():
    result = {
        "report": {},
        "unauthorized_one": True,
        "unauthorized_two": True,
    }

    with pytest.raises(
        AgentContractError,
        match=(
            "unexpected output fields: "
            "unauthorized_one, unauthorized_two"
        ),
    ):
        validate_agent_outputs("action", result)


def test_non_string_output_key_is_rejected():
    result = {
        "report": {},
        123: True,
    }

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields: 123",
    ):
        validate_agent_outputs("action", result)


def test_missing_required_output_takes_priority():
    result = {"unauthorized_field": True}

    with pytest.raises(
        AgentContractError,
        match="missing output fields: report",
    ):
        validate_agent_outputs("action", result)


def test_rejected_output_is_not_mutated():
    result = {
        "report": {},
        "unauthorized_field": True,
    }

    with pytest.raises(AgentContractError):
        validate_agent_outputs("action", result)

    assert result == {
        "report": {},
        "unauthorized_field": True,
    }
