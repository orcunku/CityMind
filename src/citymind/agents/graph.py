from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from citymind.detection.anomaly import detect_pm25_anomaly
from citymind.tools.measurement_quality import measurement_quality_tool


class CityMindState(TypedDict, total=False):
    measurement: dict[str, Any]
    monitoring_status: str
    analysis: dict[str, Any]
    decision: str
    report: dict[str, Any]


def monitoring_node(state: CityMindState) -> dict:
    measurement = state.get("measurement")

    if not isinstance(measurement, dict):
        raise ValueError(
            "A valid measurement dictionary is required"
        )

    return {
        "monitoring_status": "measurement_received"
    }


def investigation_node(state: CityMindState) -> dict:
    """
    First call the measurement quality tool,
    then perform PM2.5 anomaly detection.
    """

    measurement = state["measurement"]

    quality = measurement_quality_tool(measurement)
    analysis = detect_pm25_anomaly(measurement)

    analysis["data_status"] = quality["data_status"]
    analysis["is_fresh"] = quality["is_fresh"]

    return {
        "analysis": analysis
    }


def route_after_investigation(
    state: CityMindState,
) -> str:
    """
    Route historical and current measurements
    through different paths.
    """

    if state["analysis"]["data_status"] != "fresh":
        return "historical"

    return "current"


def historical_node(state: CityMindState) -> dict:
    """
    Historical measurements must not
    trigger live alerts.
    """

    return {
        "decision": "historical_analysis_only"
    }


def decision_node(state: CityMindState) -> dict:
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


def action_node(state: CityMindState) -> dict:
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


def build_citymind_graph():
    builder = StateGraph(CityMindState)

    builder.add_node("monitoring", monitoring_node)
    builder.add_node("investigation", investigation_node)
    builder.add_node("historical", historical_node)
    builder.add_node("decision", decision_node)
    builder.add_node("action", action_node)

    builder.add_edge(START, "monitoring")
    builder.add_edge("monitoring", "investigation")

    builder.add_conditional_edges(
        "investigation",
        route_after_investigation,
        {
            "historical": "historical",
            "current": "decision",
        },
    )

    builder.add_edge("historical", "action")
    builder.add_edge("decision", "action")
    builder.add_edge("action", END)

    return builder.compile()


citymind_graph = build_citymind_graph()
