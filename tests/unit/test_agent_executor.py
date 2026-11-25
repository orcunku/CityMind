"""
CITYMIND Agent Execution Wrapper tests.
"""

from unittest.mock import Mock

import pytest

from citymind.agents.contracts import AgentContractError
from citymind.agents.executor import execute_agent
from citymind.agents.coordinator import coordinator_node


def test_valid_agent_execution():
    state = {"measurement": {"value": 25.0}}

    expected = {
        "quality_result": {},
        "analysis": {},
    }

    agent = Mock(return_value=expected)

    result = execute_agent(
        "investigation",
        agent,
        state,
    )

    assert result == expected
    agent.assert_called_once_with(state)


def test_missing_input_prevents_execution():
    agent = Mock()

    with pytest.raises(
        AgentContractError,
        match="missing input fields",
    ):
        execute_agent("investigation", agent, {})

    agent.assert_not_called()


def test_unknown_agent_prevents_execution():
    agent = Mock()

    with pytest.raises(
        AgentContractError,
        match="Unknown agent",
    ):
        execute_agent("unknown_agent", agent, {})

    agent.assert_not_called()


@pytest.mark.parametrize(
    "state",
    [None, [], "invalid", 42],
)
def test_invalid_state_prevents_execution(state):
    agent = Mock()

    with pytest.raises(AgentContractError):
        execute_agent("coordinator", agent, state)

    agent.assert_not_called()


def test_missing_output_is_rejected():
    agent = Mock(return_value={})

    with pytest.raises(
        AgentContractError,
        match="missing output fields",
    ):
        execute_agent(
            "investigation",
            agent,
            {"measurement": {}},
        )

    agent.assert_called_once()


def test_unauthorized_output_is_rejected():
    agent = Mock(
        return_value={
            "report": {},
            "decision": "unauthorized_decision",
        },
    )

    with pytest.raises(
        AgentContractError,
        match="unexpected output fields: decision",
    ):
        execute_agent(
            "action",
            agent,
            {
                "analysis": {},
                "decision": "no_action_required",
            },
        )

    agent.assert_called_once()


@pytest.mark.parametrize(
    "result",
    [None, [], "invalid", 42],
)
def test_invalid_agent_output_type(result):
    agent = Mock(return_value=result)

    with pytest.raises(
        AgentContractError,
        match="output must be a dictionary",
    ):
        execute_agent("coordinator", agent, {})


@pytest.mark.parametrize(
    "result",
    [
        {
            "memory_status": "saved",
            "incident_id": 1,
            "vector_status": "indexed",
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
def test_memory_optional_outputs(result):
    agent = Mock(return_value=result)

    state = {
        "measurement": {},
        "analysis": {},
        "decision": "historical_analysis_only",
    }

    assert execute_agent(
        "memory",
        agent,
        state,
    ) == result


def test_agent_execution_error_is_propagated():
    agent = Mock(
        side_effect=RuntimeError("Simulated agent failure"),
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated agent failure",
    ):
        execute_agent("coordinator", agent, {})


def test_real_coordinator_rejects_missing_measurement():
    result = execute_agent(
        "coordinator",
        coordinator_node,
        {},
    )

    assert result["coordination_status"] == "rejected"
    assert result["assigned_task"] == (
        "reject_invalid_measurement"
    )
