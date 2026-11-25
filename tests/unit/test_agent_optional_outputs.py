"""
CITYMIND optional agent output contract tests.
"""

import pytest

from citymind.agents.contracts import (
    AGENT_CONTRACTS,
    AgentContractError,
    validate_agent_outputs,
)


def test_memory_optional_output_definition():
    contract = AGENT_CONTRACTS["memory"]

    assert contract.optional_outputs == (
        "vector_status",
    )


def test_other_agents_have_no_optional_outputs():
    for name, contract in AGENT_CONTRACTS.items():
        if name != "memory":
            assert contract.optional_outputs == ()


def test_required_and_optional_outputs_do_not_overlap():
    for contract in AGENT_CONTRACTS.values():
        assert not (
            set(contract.expected_outputs)
            & set(contract.optional_outputs)
        )


def test_optional_outputs_have_no_duplicates():
    for contract in AGENT_CONTRACTS.values():
        assert len(contract.optional_outputs) == len(
            set(contract.optional_outputs)
        )


@pytest.mark.parametrize(
    "result",
    [
        {
            "memory_status": "saved",
            "incident_id": 1,
            "vector_status": "indexed",
        },
        {
            "memory_status": "saved",
            "incident_id": 1,
            "vector_status": "unavailable",
        },
        {
            "memory_status": "saved",
            "incident_id": 1,
        },
        {
            "memory_status": "skipped",
            "incident_id": None,
        },
        {
            "memory_status": "unavailable",
            "incident_id": None,
        },
    ],
)
def test_valid_memory_output_scenarios(result):
    validate_agent_outputs("memory", result)


@pytest.mark.parametrize(
    "result",
    [
        {"incident_id": 1},
        {"memory_status": "saved"},
        {"vector_status": "indexed"},
    ],
)
def test_missing_required_memory_outputs(result):
    with pytest.raises(
        AgentContractError,
        match="missing output fields",
    ):
        validate_agent_outputs("memory", result)
