"""
CITYMIND Agent Contracts.

Defines the expected inputs and outputs
of deterministic workflow agents.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class AgentContract:
    name: str
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    optional_outputs: tuple[str, ...] = ()


AGENT_CONTRACTS = MappingProxyType({
    "coordinator": AgentContract(
        name="coordinator",
        required_inputs=(),
        expected_outputs=(
            "coordination_status",
            "assigned_task",
        ),
    ),
    "monitoring": AgentContract(
        name="monitoring",
        required_inputs=("measurement",),
        expected_outputs=("monitoring_status",),
    ),
    "investigation": AgentContract(
        name="investigation",
        required_inputs=("measurement",),
        expected_outputs=(
            "quality_result",
            "analysis",
        ),
    ),
    "historical": AgentContract(
        name="historical",
        required_inputs=(),
        expected_outputs=("decision",),
    ),
    "decision": AgentContract(
        name="decision",
        required_inputs=("analysis",),
        expected_outputs=("decision",),
    ),
    "action": AgentContract(
        name="action",
        required_inputs=("analysis", "decision"),
        expected_outputs=("report",),
    ),
    "memory": AgentContract(
        name="memory",
        required_inputs=(
            "measurement",
            "analysis",
            "decision",
        ),
        expected_outputs=(
            "memory_status",
            "incident_id",
        ),
        optional_outputs=("vector_status",),
    ),
})


class AgentContractError(ValueError):
    """Raised when an agent violates its contract."""


def get_agent_contract(agent_name: str) -> AgentContract:
    """Return the contract of a registered agent."""

    if agent_name not in AGENT_CONTRACTS:
        raise AgentContractError(
            f"Unknown agent: {agent_name}"
        )

    return AGENT_CONTRACTS[agent_name]


def validate_agent_inputs(
    agent_name: str,
    state: dict,
) -> None:
    """Validate required agent input fields."""

    contract = get_agent_contract(agent_name)

    if not isinstance(state, dict):
        raise AgentContractError(
            f"{agent_name}: input state must be a dictionary"
        )

    missing = [
        field
        for field in contract.required_inputs
        if field not in state
    ]

    if missing:
        raise AgentContractError(
            f"{agent_name}: missing input fields: "
            + ", ".join(missing)
        )


def validate_agent_outputs(
    agent_name: str,
    result: dict,
) -> None:
    """Validate expected agent output fields."""

    contract = get_agent_contract(agent_name)

    if not isinstance(result, dict):
        raise AgentContractError(
            f"{agent_name}: output must be a dictionary"
        )

    missing = [
        field
        for field in contract.expected_outputs
        if field not in result
    ]

    if missing:
        raise AgentContractError(
            f"{agent_name}: missing output fields: "
            + ", ".join(missing)
        )

    allowed_outputs = (
        set(contract.expected_outputs)
        | set(contract.optional_outputs)
    )

    unexpected = [
        field
        for field in result
        if field not in allowed_outputs
    ]

    if unexpected:
        raise AgentContractError(
            f"{agent_name}: unexpected output fields: "
            + ", ".join(str(field) for field in unexpected)
        )
