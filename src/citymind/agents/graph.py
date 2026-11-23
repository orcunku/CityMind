import os
from typing import Any, TypedDict

from citymind.agents.memory_node import memory_node

from langgraph.graph import END, START, StateGraph

from citymind.detection.anomaly import detect_pm25_anomaly
from citymind.tools.measurement_quality import (
    measurement_quality_tool,
)


# ============================================================
# CITYMIND - LANGGRAPH AGENT STATE
# ============================================================

class CityMindState(TypedDict, total=False):
    measurement: dict[str, Any]
    monitoring_status: str
    quality_result: dict[str, Any]
    analysis: dict[str, Any]
    decision: str
    report: dict[str, Any]
    memory_status: str
    incident_id: int | None


# ============================================================
# NODE 1 - MONITORING
# ============================================================

def monitoring_node(state: CityMindState) -> dict:
    """
    Receive and validate the measurement input.
    """

    measurement = state.get("measurement")

    if not isinstance(measurement, dict):
        raise ValueError(
            "A valid measurement dictionary is required"
        )

    return {
        "monitoring_status": "measurement_received"
    }


# ============================================================
# NODE 2 - INVESTIGATION
# ============================================================

def investigation_node(state: CityMindState) -> dict:
    """
    Call the measurement quality tool once.

    Reuse the quality result during PM2.5
    anomaly detection.

    Store both results in LangGraph State.
    """

    measurement = state["measurement"]

    quality = measurement_quality_tool(
        measurement
    )

    analysis = detect_pm25_anomaly(
        measurement,
        quality=quality,
    )

    return {
        "quality_result": quality,
        "analysis": analysis,
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_investigation(
    state: CityMindState,
) -> str:
    """
    Route measurements based on data freshness.
    """

    quality = state["quality_result"]

    if quality["data_status"] != "fresh":
        return "historical"

    return "current"


# ============================================================
# NODE 3A - HISTORICAL
# ============================================================

def historical_node(state: CityMindState) -> dict:
    """
    Handle stale measurements safely.

    Historical measurements must never
    generate live alerts.
    """

    return {
        "decision": "historical_analysis_only"
    }


# ============================================================
# NODE 3B - DECISION
# ============================================================

def decision_node(state: CityMindState) -> dict:
    """
    Evaluate current measurements.
    """

    analysis = state["analysis"]

    if analysis["eligible_for_live_alert"]:
        decision = "requires_investigation"

    elif analysis["severity"] == "review":
        decision = "monitor_closely"

    else:
        decision = "no_action_required"

    return {
        "decision": decision
    }


# ============================================================
# NODE 4 - ACTION
# ============================================================

def action_node(state: CityMindState) -> dict:
    """
    Generate a safe evaluation report.

    No real live alerts are created yet.
    """

    analysis = state["analysis"]

    report = {
        "pollutant": analysis["pollutant"],
        "value": analysis["value"],
        "unit": analysis["unit"],
        "severity": analysis["severity"],
        "data_status": analysis["data_status"],
        "decision": state["decision"],
        "live_alert_created": False,
        "message": (
            "Historical data: no live alert permitted."
            if analysis["data_status"] != "fresh"
            else "Measurement evaluated; further checks may be needed."
        ),
    }

    return {
        "report": report
    }


def persist_memory_node(state: CityMindState) -> dict:
    """
    Persist the completed analysis without creating alerts.
    """
    db_path = os.getenv(
        "CITYMIND_DB_PATH",
        "data/citymind_incidents.db",
    )

    chroma_path = os.getenv(
        "CITYMIND_CHROMA_PATH",
        "chroma",
    )

    return memory_node(
        state,
        db_path=db_path,
        chroma_path=chroma_path,
    )


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_citymind_graph():
    """
    Build and compile the CITYMIND agent graph.
    """

    builder = StateGraph(CityMindState)

    # Register nodes
    builder.add_node(
        "monitoring",
        monitoring_node,
    )

    builder.add_node(
        "investigation",
        investigation_node,
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

    builder.add_node(
        "memory",
        persist_memory_node,
    )

    # Entry point
    builder.add_edge(
        START,
        "monitoring",
    )

    builder.add_edge(
        "monitoring",
        "investigation",
    )

    # Conditional routing
    builder.add_conditional_edges(
        "investigation",
        route_after_investigation,
        {
            "historical": "historical",
            "current": "decision",
        },
    )

    # Both paths lead to action
    builder.add_edge(
        "historical",
        "action",
    )

    builder.add_edge(
        "decision",
        "action",
    )

    # Finish workflow
    builder.add_edge(
        "action",
        "memory",
    )

    builder.add_edge(
        "memory",
        END,
    )

    return builder.compile()


# ============================================================
# COMPILED CITYMIND AGENT
# ============================================================

citymind_graph = build_citymind_graph()

