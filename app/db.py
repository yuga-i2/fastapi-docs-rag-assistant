"""
Lightweight SQLite logging for every question asked and the feedback it
receives. This is the data source for the monitoring dashboard.

Using SQLite keeps the project runnable with zero extra infrastructure;
docker-compose.yml also provides a Postgres option if you want a "real"
DB for a multi-user deployment (see README).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "monitoring.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    response_time_ms REAL NOT NULL,
    feedback INTEGER  -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_interaction(question: str, answer: str, retrieval_method: str,
                     sources_json: str, response_time_ms: float) -> str:
    init_db()
    interaction_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO interactions "
            "(id, timestamp, question, answer, retrieval_method, sources_json, response_time_ms, feedback) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (interaction_id, time.time(), question, answer, retrieval_method,
             sources_json, response_time_ms),
        )
        conn.commit()
    return interaction_id


def record_feedback(interaction_id: str, feedback: int):
    init_db()
    with get_conn() as conn:
        conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (feedback, interaction_id),
        )
        conn.commit()


def fetch_all() -> list[dict]:
    init_db()
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM interactions ORDER BY timestamp DESC").fetchall()
        return [dict(r) for r in rows]
