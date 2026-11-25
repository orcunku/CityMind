"""
CITYMIND Agent Contracts tests.
"""

from dataclasses import FrozenInstanceError

import pytest

from citymind.agents.contracts import (
    AgentContract,
    AGENT_CONTRACTS,
)


def test_all_agent_contracts_exist():
    assert set(AGENT_CONTRACTS) == {
        "coordinator",
        "monitoring",
        "investigation",
        "historical",
        "decision",
        "action",
        "memory",
    }


@pytest.mark.parametrize(
    "name",
    tuple(AGENT_CONTRACTS),
)
def test_contract_names_and_fields(name):
    contract = AGENT_CONTRACTS[name]

    assert isinstance(contract, AgentContract)
    assert contract.name == name

    assert isinstance(contract.required_inputs, tuple)
    assert isinstance(contract.expected_outputs, tuple)

    assert len(contract.expected_outputs) > 0

    assert len(contract.required_inputs) == len(
        set(contract.required_inputs)
    )

    assert len(contract.expected_outputs) == len(
        set(contract.expected_outputs)
    )


def test_contract_is_immutable():
    contract = AGENT_CONTRACTS["coordinator"]

    with pytest.raises(FrozenInstanceError):
        contract.name = "modified"


def test_coordinator_accepts_missing_measurement():
    contract = AGENT_CONTRACTS["coordinator"]

    assert contract.required_inputs == ()

    assert contract.expected_outputs == (
        "coordination_status",
        "assigned_task",
    )


def test_investigation_contract():
    contract = AGENT_CONTRACTS["investigation"]

    assert contract.required_inputs == ("measurement",)

    assert contract.expected_outputs == (
        "quality_result",
        "analysis",
    )


def test_action_contract():
    contract = AGENT_CONTRACTS["action"]

    assert contract.required_inputs == (
        "analysis",
        "decision",
    )

    assert contract.expected_outputs == ("report",)


def test_memory_contract():
    contract = AGENT_CONTRACTS["memory"]

    assert contract.required_inputs == (
        "measurement",
        "analysis",
        "decision",
    )

    assert contract.expected_outputs == (
        "memory_status",
        "incident_id",
    )

    assert contract.optional_outputs == (
        "vector_status",
    )
