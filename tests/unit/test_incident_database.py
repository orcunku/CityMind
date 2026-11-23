"""
Tests for CITYMIND SQLite incident database initialization.
"""

import sqlite3

from citymind.memory.database import initialize_database


def test_initialize_database_creates_incidents_table(tmp_path):
    db_path = tmp_path / "incidents.db"

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = connection.execute(
            "PRAGMA table_info(incidents)"
        ).fetchall()

    column_names = [column[1] for column in columns]

    assert column_names == [
        "id",
        "measurement_value",
        "measured_at",
        "data_status",
        "severity",
        "decision",
        "created_at",
    ]


def test_initialize_database_is_idempotent(tmp_path):
    db_path = tmp_path / "incidents.db"

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO incidents (
                measurement_value,
                measured_at,
                data_status,
                severity,
                decision
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                34.19,
                "2022-10-25T12:00:00Z",
                "stale",
                "high",
                "historical_analysis_only",
            ),
        )
        connection.commit()

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

    assert count == 1
