"""
CITYMIND agent state isolation security tests.
"""

from copy import deepcopy

import pytest

from citymind.agents.contracts import AgentContractError
from citymind.agents.executor import execute_agent


def sample_state():
    return {
        "measurement": {
            "value": 25.0,
            "metadata": {
                "station": "Alsancak",
                "tags": ["pm25"],
            },
        },
        "decision": "historical_analysis_only",
    }


def valid_output():
    return {
        "quality_result": {},
        "analysis": {},
    }


def test_agent_receives_separate_state_object():
    state = sample_state()

    def agent(agent_state):
        assert agent_state is not state
        assert agent_state == state
        return valid_output()

    execute_agent("investigation", agent, state)


def test_top_level_state_is_protected():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        agent_state["decision"] = "unauthorized"
        return valid_output()

    execute_agent("investigation", agent, state)

    assert state == original


def test_nested_dictionary_is_protected():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        agent_state["measurement"]["value"] = 999.0
        agent_state["measurement"]["metadata"]["station"] = (
            "modified"
        )
        return valid_output()

    execute_agent("investigation", agent, state)

    assert state == original


def test_nested_list_is_protected():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        agent_state["measurement"]["metadata"]["tags"].append(
            "unauthorized"
        )
        return valid_output()

    execute_agent("investigation", agent, state)

    assert state == original


def test_state_field_injection_is_isolated():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        agent_state["unauthorized_field"] = True
        return valid_output()

    execute_agent("investigation", agent, state)

    assert state == original


def test_state_field_deletion_is_isolated():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        del agent_state["measurement"]
        return valid_output()

    execute_agent("investigation", agent, state)

    assert state == original


def test_state_is_protected_after_output_rejection():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        agent_state["measurement"]["value"] = 999.0

        return {
            **valid_output(),
            "unauthorized_field": True,
        }

    with pytest.raises(AgentContractError):
        execute_agent("investigation", agent, state)

    assert state == original


def test_state_is_protected_after_agent_failure():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        agent_state["measurement"]["value"] = 999.0
        raise RuntimeError("Simulated agent failure")

    with pytest.raises(
        RuntimeError,
        match="Simulated agent failure",
    ):
        execute_agent("investigation", agent, state)

    assert state == original


def test_sequential_agents_receive_isolated_state():
    state = sample_state()

    def first_agent(agent_state):
        agent_state["measurement"]["value"] = 999.0
        return valid_output()

    def second_agent(agent_state):
        assert agent_state["measurement"]["value"] == 25.0
        return valid_output()

    execute_agent("investigation", first_agent, state)
    execute_agent("investigation", second_agent, state)

    assert state["measurement"]["value"] == 25.0


def test_returned_nested_data_is_detached_from_input():
    state = sample_state()
    original = deepcopy(state)

    def agent(agent_state):
        return {
            "quality_result": {},
            "analysis": agent_state["measurement"],
        }

    result = execute_agent(
        "investigation",
        agent,
        state,
    )

    result["analysis"]["value"] = 999.0

    assert state == original
