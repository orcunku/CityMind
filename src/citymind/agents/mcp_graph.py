"""
CITYMIND MCP-backed LangGraph workflow.

Uses MCP for measurement quality investigation.

The original synchronous graph remains unchanged.
No live alerts are sent.
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


def build_citymind_mcp_graph():
    """
    Build the asynchronous MCP-backed
    CityMind monitoring workflow.
    """

    builder = StateGraph(CityMindState)

    builder.add_node(
        "monitoring",
        monitoring_node,
    )

    builder.add_node(
        "investigation",
        mcp_investigation_node,
    )

    builder.add_node(
        "historical",
        historical_node,
    )

    builder.add_node(
        "decision",
        decision_node,
    )

    builder.add_node(
        "action",
        action_node,
    )

    builder.add_edge(
        START,
        "monitoring",
    )

    builder.add_edge(
        "monitoring",
        "investigation",
    )

    builder.add_conditional_edges(
        "investigation",
        route_after_investigation,
        {
            "historical": "historical",
            "current": "decision",
        },
    )

    builder.add_edge(
        "historical",
        "action",
    )

    builder.add_edge(
        "decision",
        "action",
    )

    builder.add_edge(
        "action",
        END,
    )

    return builder.compile()


citymind_mcp_graph = build_citymind_mcp_graph()
