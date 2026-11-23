"""
CITYMIND pytest configuration.

Isolate SQLite incident memory for every test.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_incident_database(tmp_path, monkeypatch):
    """
    Prevent tests from writing to the project's
    default SQLite incident database.
    """
    db_path = tmp_path / "citymind_test_incidents.db"

    monkeypatch.setenv(
        "CITYMIND_DB_PATH",
        str(db_path),
    )

    monkeypatch.setenv(
        "CITYMIND_CHROMA_PATH",
        str(tmp_path / "citymind_test_chroma"),
    )
