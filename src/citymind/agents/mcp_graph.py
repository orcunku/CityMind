"""
CITYMIND MCP-backed LangGraph workflow.

MCP connection failures produce a safe report.
The original synchronous graph is unchanged.
"""

from langgraph.graph import END, START, StateGraph

from citymind.agents.graph import (
    CityMindState,
    monitoring_node,
    route_after_investigation,
    historical_node,
    decision_node,
    action_node,
)
from citymind.agents.mcp_investigation import (
    mcp_investigation_node,
)
from citymind.mcp.errors import MCPConnectionError, MCPInvalidResponseError


async def safe_mcp_investigation_node(state):
    """
    Handle MCP connection failures without
    producing a misleading successful analysis.
    """
    try:
        return await mcp_investigation_node(state)
    except (ConnectionError, MCPConnectionError):
        error_type = "mcp_connection_error"
    except MCPInvalidResponseError:
        error_type = "mcp_invalid_response"
    return {
        "quality_result": {
            "data_status": "unavailable",
            "is_fresh": False,
            "eligible_for_live_evaluation": False,
        },
        "analysis": {
            "data_status": "unavailable",
            "eligible_for_live_alert": False,
        },
        "decision": "investigation_unavailable",
        "report": {
            "data_status": "unavailable",
            "error_type": error_type,
            "live_alert_created": False,
        },
    }


def route_after_mcp_investigation(state):
    """
    Route unavailable MCP results to a safe
    error report instead of normal decisions.
    """
    if (
        state.get("quality_result", {}).get("data_status")
        == "unavailable"
    ):
        return "unavailable"

    return route_after_investigation(state)


def mcp_failure_report_node(state):
    """
    Create a report without sending alerts.
    """
    return {
        "decision": "investigation_unavailable",
        "report": {
            "data_status": "unavailable",
            "error_type": state["report"]["error_type"],
            "live_alert_created": False,
        },
    }


def build_citymind_mcp_graph():
    builder = StateGraph(CityMindState)

    builder.add_node("monitoring", monitoring_node)
    builder.add_node(
        "investigation",
        safe_mcp_investigation_node,
    )
    builder.add_node("historical", historical_node)
    builder.add_node("decision", decision_node)
    builder.add_node("action", action_node)
    builder.add_node(
        "mcp_failure_report",
        mcp_failure_report_node,
    )

    builder.add_edge(START, "monitoring")
    builder.add_edge("monitoring", "investigation")

    builder.add_conditional_edges(
        "investigation",
        route_after_mcp_investigation,
        {
            "historical": "historical",
            "current": "decision",
            "unavailable": "mcp_failure_report",
        },
    )

    builder.add_edge("historical", "action")
    builder.add_edge("decision", "action")
    builder.add_edge("action", END)

    builder.add_edge("mcp_failure_report", END)

    return builder.compile()


citymind_mcp_graph = build_citymind_mcp_graph()
