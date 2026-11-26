"""
CITYMIND persistent incident memory.

Stores analysis incidents in a local SQLite database.
No live alerts are created by this module.
"""

import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data/citymind_incidents.db")


def initialize_database(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """
    Create the incident database and table if missing.
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_value REAL NOT NULL,
                measured_at TEXT NOT NULL,
                data_status TEXT NOT NULL,
                severity TEXT NOT NULL,
                decision TEXT NOT NULL,
                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_incident(
    measurement_value: float,
    measured_at: str,
    data_status: str,
    severity: str,
    decision: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    Persist an incident and return its database ID.

    This function does not create or send alerts.
    """
    import math

    if (
        isinstance(measurement_value, bool)
        or not isinstance(measurement_value, (int, float))
    ):
        raise ValueError("Invalid measurement value")

    try:
        measurement_value = float(measurement_value)
    except (OverflowError, ValueError, TypeError):
        raise ValueError("Invalid measurement value") from None

    if (
        not math.isfinite(measurement_value)
        or measurement_value < 0
    ):
        raise ValueError("Invalid measurement value")

    for field_name, value in {
        "measured_at": measured_at,
        "data_status": data_status,
        "severity": severity,
        "decision": decision,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid {field_name}")

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
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
                measurement_value,
                measured_at,
                data_status,
                severity,
                decision,
            ),
        )
        incident_id = cursor.lastrowid

    return incident_id


def get_incident(
    incident_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    """
    Retrieve an incident by ID.

    Returns None when the incident does not exist.
    Does not modify the database.
    """
    if (
        isinstance(incident_id, bool)
        or not isinstance(incident_id, int)
        or not 1 <= incident_id <= 2**63 - 1
    ):
        raise ValueError("Invalid incident ID")

    db_path = Path(db_path)

    if not db_path.is_file():
        return None

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                id,
                measurement_value,
                measured_at,
                data_status,
                severity,
                decision,
                created_at
            FROM incidents
            WHERE id = ?
            """,
            (incident_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def list_recent_incidents(
    limit: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Return the most recently inserted incidents.

    This is read-only and does not create alerts.
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError("Invalid incident limit")

    db_path = Path(db_path)

    if not db_path.is_file():
        return []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                measurement_value,
                measured_at,
                data_status,
                severity,
                decision,
                created_at
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
