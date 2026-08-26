"""
history.py
----------
SQLite-based recent query history for Smart FAQ Bot.

Stores the user's recent questions permanently in SQLite so that
history remains available even after Streamlit refreshes/restarts.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# DATABASE LOCATION

BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_DB = BASE_DIR / "history.db"

# DATABASE CONNECTION

def get_connection():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(str(HISTORY_DB))

# INITIALIZE DATABASE

def init_history_db():
    """Create the history table if it does not already exist."""

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()

# ADD QUERY

def add_query(question: str):
    """
    Add a question to recent history.

    If the same question already exists, the old entry is removed
    and the new one becomes the most recent entry.

    Only the latest 10 queries are retained.
    """

    if not question or not question.strip():
        return

    question = question.strip()

    conn = get_connection()
    cursor = conn.cursor()

    # Remove duplicate question
    cursor.execute(
        """
        DELETE FROM query_history
        WHERE LOWER(question) = LOWER(?)
        """,
        (question,),
    )

    # Add the question as the newest entry
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO query_history
        (question, timestamp)
        VALUES (?, ?)
        """,
        (
            question,
            timestamp,
        ),
    )

    # Keep only the latest 10
    cursor.execute(
        """
        DELETE FROM query_history
        WHERE id NOT IN (
            SELECT id
            FROM query_history
            ORDER BY id DESC
            LIMIT 10
        )
        """
    )

    conn.commit()
    conn.close()


# GET RECENT QUERIES

def get_recent_queries(limit=5):
    """
    Get the most recent questions.

    Returns a list of dictionaries:

    [
        {
            "id": 1,
            "question": "...",
            "timestamp": "2026-08-22 01:20:30"
        }
    ]
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, question, timestamp
        FROM query_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    conn.close()

    history = []

    for row in rows:
        history.append(
            {
                "id": row[0],
                "question": row[1],
                "timestamp": row[2],
            }
        )

    return history

# CLEAR HISTORY

def clear_history():
    """Delete all stored query history."""

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM query_history"
    )

    conn.commit()
    conn.close()

# INITIALIZE DATABASE

init_history_db()