"""
CITYMIND Agent Execution Wrapper.

Validates agent inputs and outputs before
returning a state update to the workflow.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from citymind.agents.contracts import (
    validate_agent_inputs,
    validate_agent_outputs,
)

def execute_agent(
    agent_name: str,
    agent_function: Callable[
        [dict[str, Any]],
        dict[str, Any],
    ],
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute an agent with contract validation.

    Invalid inputs prevent agent execution.
    Invalid outputs prevent the result from
    being returned to the workflow.
    """

    validate_agent_inputs(agent_name, state)

    isolated_state = deepcopy(state)

    result = agent_function(isolated_state)

    validate_agent_outputs(agent_name, result)

    return result
