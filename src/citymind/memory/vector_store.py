"""
CITYMIND ChromaDB incident vector memory.

Stores historical incident representations locally.
Uses deterministic numeric features for the prototype.
"""

import math
from pathlib import Path

import chromadb


COLLECTION_NAME = "citymind_incidents"


def incident_embedding(
    measurement_value: float,
    data_status: str,
    severity: str,
) -> list[float]:
    """
    Build a deterministic feature vector.

    This is not a semantic language-model embedding.
    """

    if (
        isinstance(measurement_value, bool)
        or not isinstance(measurement_value, (int, float))
        or not math.isfinite(measurement_value)
        or measurement_value < 0
    ):
        raise ValueError(
            "Measurement value must be finite and non-negative"
        )

    if data_status not in ("fresh", "stale"):
        raise ValueError("Invalid data status")

    if severity not in ("normal", "review", "high"):
        raise ValueError("Invalid severity")

    return [
        min(float(measurement_value) / 100.0, 10.0),
        1.0 if data_status == "fresh" else 0.0,
        1.0 if severity == "normal" else 0.0,
        1.0 if severity == "review" else 0.0,
        1.0 if severity == "high" else 0.0,
    ]


def get_incident_collection(
    persist_directory: str | Path = "chroma",
):
    """
    Open or create the persistent incident collection.
    """

    client = chromadb.PersistentClient(
        path=str(persist_directory),
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "l2",
        },
    )


def save_incident_vector(
    incident_id: int,
    measurement_value: float,
    data_status: str,
    severity: str,
    decision: str,
    persist_directory: str | Path = "chroma",
) -> str:
    """
    Save a SQLite incident representation in ChromaDB.
    """

    if isinstance(incident_id, bool) or not isinstance(
        incident_id, int
    ) or incident_id <= 0:
        raise ValueError("Invalid incident ID")

    if not isinstance(decision, str) or not decision.strip():
        raise ValueError("Invalid decision")

    embedding = incident_embedding(
        measurement_value,
        data_status,
        severity,
    )

    collection = get_incident_collection(
        persist_directory=persist_directory,
    )

    vector_id = f"incident-{incident_id}"

    document = (
        f"PM2.5 measurement: {measurement_value} µg/m³. "
        f"Data status: {data_status}. "
        f"Severity: {severity}. "
        f"Decision: {decision}."
    )

    collection.upsert(
        ids=[vector_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[{
            "incident_id": incident_id,
            "measurement_value": float(measurement_value),
            "data_status": data_status,
            "severity": severity,
            "decision": decision,
        }],
    )

    return vector_id


def search_similar_incidents(
    measurement_value: float,
    data_status: str,
    severity: str,
    limit: int = 5,
    persist_directory: str | Path = "chroma",
) -> list[dict]:
    """
    Find historical incidents by numeric feature similarity.

    Lower distance means a closer feature-vector match.
    This is not semantic text retrieval.
    """

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError("Limit must be between 1 and 100")

    query_vector = incident_embedding(
        measurement_value=measurement_value,
        data_status=data_status,
        severity=severity,
    )

    collection = get_incident_collection(
        persist_directory=persist_directory,
    )

    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(limit, count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for vector_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "vector_id": vector_id,
            "document": document,
            "metadata": metadata,
            "distance": float(distance),
        })

    return matches
