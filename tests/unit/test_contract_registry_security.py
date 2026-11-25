"""
CITYMIND immutable contract registry security tests.
"""

from types import MappingProxyType

import pytest

from citymind.agents.contracts import (
    AGENT_CONTRACTS,
    AgentContract,
    get_agent_contract,
)


def test_registry_is_read_only_mapping():
    assert isinstance(
        AGENT_CONTRACTS,
        MappingProxyType,
    )


def test_new_contract_cannot_be_added():
    unauthorized = AgentContract(
        name="unauthorized",
        required_inputs=(),
        expected_outputs=("report",),
    )

    with pytest.raises(TypeError):
        AGENT_CONTRACTS["unauthorized"] = unauthorized

    assert "unauthorized" not in AGENT_CONTRACTS


def test_existing_contract_cannot_be_replaced():
    original = get_agent_contract("coordinator")

    unauthorized = AgentContract(
        name="coordinator",
        required_inputs=(),
        expected_outputs=("report", "decision"),
    )

    with pytest.raises(TypeError):
        AGENT_CONTRACTS["coordinator"] = unauthorized

    assert get_agent_contract("coordinator") is original


def test_existing_contract_cannot_be_deleted():
    original = get_agent_contract("investigation")

    with pytest.raises(TypeError):
        del AGENT_CONTRACTS["investigation"]

    assert get_agent_contract("investigation") is original


def test_registry_cannot_be_cleared():
    with pytest.raises(AttributeError):
        AGENT_CONTRACTS.clear()

    assert len(AGENT_CONTRACTS) == 7


def test_registry_cannot_be_updated():
    with pytest.raises(AttributeError):
        AGENT_CONTRACTS.update({})

    assert len(AGENT_CONTRACTS) == 7


@pytest.mark.parametrize(
    "agent_name",
    [
        "coordinator",
        "monitoring",
        "investigation",
        "historical",
        "decision",
        "action",
        "memory",
    ],
)
def test_registered_contracts_remain_readable(agent_name):
    contract = get_agent_contract(agent_name)

    assert isinstance(contract, AgentContract)
    assert contract.name == agent_name
    assert contract is AGENT_CONTRACTS[agent_name]
