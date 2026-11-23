"""
Tests for SQLite-to-ChromaDB incident indexing.
"""

import pytest

from citymind.memory.database import (
    get_incident,
    initialize_database,
    save_incident,
)
from citymind.memory.indexer import index_sqlite_incident
from citymind.memory.vector_store import (
    get_incident_collection,
    search_similar_incidents,
)


@pytest.fixture
def memory_paths(tmp_path):
    db_path = tmp_path / "incidents.db"
    chroma_path = tmp_path / "chroma"

    initialize_database(db_path=db_path)

    return db_path, chroma_path


def create_incident(db_path, value=34.19):
    return save_incident(
        measurement_value=value,
        measured_at="2022-08-01T12:00:00Z",
        data_status="stale",
        severity="review",
        decision="historical_analysis_only",
        db_path=db_path,
    )


def test_sqlite_incident_is_indexed(memory_paths):
    db_path, chroma_path = memory_paths

    incident_id = create_incident(db_path)

    result = index_sqlite_incident(
        incident_id=incident_id,
        db_path=db_path,
        persist_directory=chroma_path,
    )

    assert result == {
        "index_status": "indexed",
        "vector_id": f"incident-{incident_id}",
    }

    collection = get_incident_collection(chroma_path)

    assert collection.count() == 1


def test_sqlite_chroma_data_consistency(memory_paths):
    db_path, chroma_path = memory_paths

    incident_id = create_incident(db_path)

    index_sqlite_incident(
        incident_id,
        db_path,
        chroma_path,
    )

    sqlite_record = get_incident(
        incident_id,
        db_path=db_path,
    )

    collection = get_incident_collection(chroma_path)

    result = collection.get(
        ids=[f"incident-{incident_id}"],
        include=["metadatas"],
    )

    metadata = result["metadatas"][0]

    assert metadata["incident_id"] == sqlite_record["id"]
    assert (
        metadata["measurement_value"]
        == sqlite_record["measurement_value"]
    )
    assert metadata["data_status"] == sqlite_record["data_status"]
    assert metadata["severity"] == sqlite_record["severity"]
    assert metadata["decision"] == sqlite_record["decision"]


def test_indexed_incident_is_searchable(memory_paths):
    db_path, chroma_path = memory_paths

    incident_id = create_incident(db_path)

    index_sqlite_incident(
        incident_id,
        db_path,
        chroma_path,
    )

    matches = search_similar_incidents(
        measurement_value=35.0,
        data_status="stale",
        severity="review",
        persist_directory=chroma_path,
    )

    assert len(matches) == 1
    assert (
        matches[0]["metadata"]["incident_id"]
        == incident_id
    )


def test_missing_incident_is_not_indexed(memory_paths):
    db_path, chroma_path = memory_paths

    result = index_sqlite_incident(
        incident_id=99999,
        db_path=db_path,
        persist_directory=chroma_path,
    )

    assert result == {
        "index_status": "not_found",
        "vector_id": None,
    }


def test_reindexing_does_not_duplicate(memory_paths):
    db_path, chroma_path = memory_paths

    incident_id = create_incident(db_path)

    for _ in range(2):
        result = index_sqlite_incident(
            incident_id,
            db_path,
            chroma_path,
        )
        assert result["index_status"] == "indexed"

    collection = get_incident_collection(chroma_path)

    assert collection.count() == 1


def test_multiple_incidents_keep_unique_ids(memory_paths):
    db_path, chroma_path = memory_paths

    first_id = create_incident(db_path, value=20.0)
    second_id = create_incident(db_path, value=34.0)

    for incident_id in (first_id, second_id):
        result = index_sqlite_incident(
            incident_id,
            db_path,
            chroma_path,
        )
        assert result["index_status"] == "indexed"

    collection = get_incident_collection(chroma_path)

    assert collection.count() == 2

    stored = collection.get()

    assert set(stored["ids"]) == {
        f"incident-{first_id}",
        f"incident-{second_id}",
    }
