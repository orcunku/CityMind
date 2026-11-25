"""
CITYMIND LangGraph agent contract integration tests.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from citymind.agents import graph
from citymind.agents.contracts import AgentContractError


def make_measurement(age_hours=1):
    timestamp = (
        datetime.now(timezone.utc)
        - timedelta(hours=age_hours)
    ).isoformat()

    return {
        "value": 45.0,
        "period": {
            "datetimeTo": {
                "utc": timestamp,
            },
        },
    }


def called_agents(executor):
    return [
        call.args[0]
        for call in executor.call_args_list
    ]


def test_fresh_workflow_uses_execution_wrapper():
    with patch.object(
        graph,
        "execute_agent",
        wraps=graph.execute_agent,
    ) as executor:
        result = graph.citymind_graph.invoke({
            "measurement": make_measurement(),
        })

    assert called_agents(executor) == [
        "coordinator",
        "monitoring",
        "investigation",
        "decision",
        "action",
        "memory",
    ]

    assert result["decision"] == "requires_investigation"
    assert result["report"]["live_alert_created"] is False


def test_historical_workflow_uses_execution_wrapper():
    with patch.object(
        graph,
        "execute_agent",
        wraps=graph.execute_agent,
    ) as executor:
        result = graph.citymind_graph.invoke({
            "measurement": make_measurement(72),
        })

    assert called_agents(executor) == [
        "coordinator",
        "monitoring",
        "investigation",
        "historical",
        "action",
        "memory",
    ]

    assert result["decision"] == "historical_analysis_only"
    assert result["report"]["live_alert_created"] is False


def test_rejected_workflow_stops_after_coordinator():
    with patch.object(
        graph,
        "execute_agent",
        wraps=graph.execute_agent,
    ) as executor:
        result = graph.citymind_graph.invoke({})

    assert called_agents(executor) == ["coordinator"]
    assert result["coordination_status"] == "rejected"
    assert "report" not in result
    assert "incident_id" not in result


@pytest.mark.parametrize(
    "agent_name, function_name, invalid_output",
    [
        (
            "coordinator",
            "coordinator_node",
            {
                "coordination_status": "assigned",
                "assigned_task": "investigate_measurement",
                "decision": "unauthorized",
            },
        ),
        (
            "investigation",
            "investigation_node",
            {
                "quality_result": {},
                "analysis": {},
                "report": {},
            },
        ),
        (
            "action",
            "action_node",
            {
                "report": {},
                "decision": "unauthorized",
            },
        ),
    ],
)
def test_contract_violation_stops_workflow(
    agent_name,
    function_name,
    invalid_output,
):
    with (
        patch.object(
            graph,
            function_name,
            return_value=invalid_output,
        ),
        patch.object(
            graph,
            "execute_agent",
            wraps=graph.execute_agent,
        ) as executor,
        patch.object(
            graph,
            "persist_memory_node",
        ) as memory,
    ):
        with pytest.raises(
            AgentContractError,
            match="unexpected output fields",
        ):
            graph.citymind_graph.invoke({
                "measurement": make_measurement(),
            })

        memory.assert_not_called()

    assert called_agents(executor)[-1] == agent_name


def test_missing_investigation_output_stops_workflow():
    with (
        patch.object(
            graph,
            "investigation_node",
            return_value={"analysis": {}},
        ),
        patch.object(graph, "action_node") as action,
        patch.object(graph, "persist_memory_node") as memory,
    ):
        with pytest.raises(
            AgentContractError,
            match="missing output fields",
        ):
            graph.citymind_graph.invoke({
                "measurement": make_measurement(),
            })

        action.assert_not_called()
        memory.assert_not_called()


def test_graph_protects_measurement_from_agent_mutation():
    measurement = make_measurement()
    original_investigation = graph.investigation_node

    def mutating_investigation(state):
        result = original_investigation(state)
        state["measurement"]["value"] = 999.0
        return result

    with patch.object(
        graph,
        "investigation_node",
        side_effect=mutating_investigation,
    ):
        result = graph.citymind_graph.invoke({
            "measurement": measurement,
        })

    assert measurement["value"] == 45.0
    assert result["measurement"]["value"] == 45.0
    assert result["analysis"]["value"] == 45.0
    assert result["report"]["value"] == 45.0
