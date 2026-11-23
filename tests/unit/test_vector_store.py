"""
Tests for CITYMIND ChromaDB vector memory.
"""

import pytest

from citymind.memory.vector_store import (
    get_incident_collection,
    incident_embedding,
    save_incident_vector,
    search_similar_incidents,
)


def test_incident_embedding():
    vector = incident_embedding(
        34.19,
        "stale",
        "review",
    )

    assert vector == [
        0.3419,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


@pytest.mark.parametrize(
    "value",
    [-1, float("nan"), float("inf"), True],
)
def test_invalid_measurement_value(value):
    with pytest.raises(ValueError):
        incident_embedding(
            value,
            "stale",
            "review",
        )


def test_invalid_status_and_severity():
    with pytest.raises(ValueError):
        incident_embedding(20, "unknown", "review")

    with pytest.raises(ValueError):
        incident_embedding(20, "stale", "unknown")


def test_persistent_storage(tmp_path):
    chroma_path = tmp_path / "chroma"

    vector_id = save_incident_vector(
        incident_id=1,
        measurement_value=34.19,
        data_status="stale",
        severity="review",
        decision="historical_analysis_only",
        persist_directory=chroma_path,
    )

    assert vector_id == "incident-1"

    collection = get_incident_collection(
        persist_directory=chroma_path,
    )

    result = collection.get(
        ids=["incident-1"],
        include=["metadatas"],
    )

    assert result["ids"] == ["incident-1"]
    assert result["metadatas"][0]["incident_id"] == 1
    assert result["metadatas"][0]["data_status"] == "stale"


def test_upsert_prevents_duplicates(tmp_path):
    chroma_path = tmp_path / "chroma"

    for _ in range(2):
        save_incident_vector(
            incident_id=1,
            measurement_value=34.19,
            data_status="stale",
            severity="review",
            decision="historical_analysis_only",
            persist_directory=chroma_path,
        )

    collection = get_incident_collection(
        persist_directory=chroma_path,
    )

    assert collection.count() == 1


def test_similarity_search(tmp_path):
    chroma_path = tmp_path / "chroma"

    incidents = [
        (1, 12.0, "stale", "normal"),
        (2, 34.0, "stale", "review"),
        (3, 85.0, "stale", "high"),
    ]

    for incident_id, value, status, severity in incidents:
        save_incident_vector(
            incident_id=incident_id,
            measurement_value=value,
            data_status=status,
            severity=severity,
            decision="historical_analysis_only",
            persist_directory=chroma_path,
        )

    matches = search_similar_incidents(
        measurement_value=35.0,
        data_status="stale",
        severity="review",
        limit=3,
        persist_directory=chroma_path,
    )

    assert len(matches) == 3
    assert matches[0]["metadata"]["incident_id"] == 2

    distances = [match["distance"] for match in matches]

    assert distances == sorted(distances)


def test_empty_collection(tmp_path):
    matches = search_similar_incidents(
        measurement_value=35.0,
        data_status="stale",
        severity="review",
        persist_directory=tmp_path / "chroma",
    )

    assert matches == []


@pytest.mark.parametrize(
    "limit",
    [0, -1, 101, True, 1.5],
)
def test_invalid_search_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        search_similar_incidents(
            measurement_value=35.0,
            data_status="stale",
            severity="review",
            limit=limit,
            persist_directory=tmp_path / "chroma",
        )


def test_invalid_incident_id(tmp_path):
    with pytest.raises(ValueError):
        save_incident_vector(
            incident_id=0,
            measurement_value=35.0,
            data_status="stale",
            severity="review",
            decision="historical_analysis_only",
            persist_directory=tmp_path / "chroma",
        )
