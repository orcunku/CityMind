from typing import Any, TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from citymind.detection.anomaly import (
    detect_pm25_anomaly,
)


class CityMindState(TypedDict, total=False):
    measurement: dict[str, Any]
    monitoring_status: str
    analysis: dict[str, Any]
    decision: str
    report: dict[str, Any]


def monitoring_node(state: CityMindState) -> dict:
    """Gelen ölçümü kontrol eder."""

    measurement = state.get("measurement")

    if not isinstance(measurement, dict):
        raise ValueError(
            "A valid measurement dictionary is required"
        )

    return {
        "monitoring_status": "measurement_received"
    }


def investigation_node(state: CityMindState) -> dict:
    """Veri kalitesini ve anomaliyi analiz eder."""

    analysis = detect_pm25_anomaly(
        state["measurement"]
    )

    return {
        "analysis": analysis
    }


def decision_node(state: CityMindState) -> dict:
    """Analiz sonucuna göre karar oluşturur."""

    analysis = state["analysis"]

    if analysis["data_status"] != "fresh":
        decision = "historical_analysis_only"

    elif analysis["eligible_for_live_alert"]:
        decision = "requires_investigation"

    elif analysis["severity"] == "review":
        decision = "monitor_closely"

    else:
        decision = "no_action_required"

    return {
        "decision": decision
    }


def action_node(state: CityMindState) -> dict:
    """Kararı yapılandırılmış rapora dönüştürür."""

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
    """CityMind LangGraph iş akışını oluşturur."""

    builder = StateGraph(CityMindState)

    builder.add_node(
        "monitoring",
        monitoring_node,
    )

    builder.add_node(
        "investigation",
        investigation_node,
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

    builder.add_edge(
        "investigation",
        "decision",
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


citymind_graph = build_citymind_graph()
