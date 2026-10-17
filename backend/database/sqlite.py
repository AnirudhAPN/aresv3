"""SQLite schema initialization for ARES."""
from __future__ import annotations

import sqlite3

from config import settings


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ResearchSession (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'live',
                current_agent TEXT,
                completed_agents_json TEXT,
                acs_scores_json TEXT,
                acs_components_json TEXT,
                retry_counts_json TEXT,
                parse_failures_json TEXT,
                flags_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS AgentOutputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                output_json TEXT,
                parse_failed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES ResearchSession(session_id)
            );

            CREATE TABLE IF NOT EXISTS Metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                condition TEXT NOT NULL DEFAULT 'adaptive',
                topic TEXT,
                confidence REAL,
                evidence_quality REAL,
                task_completion REAL,
                acs REAL,
                retries INTEGER DEFAULT 0,
                execution_time REAL,
                FOREIGN KEY(session_id) REFERENCES ResearchSession(session_id)
            );

            CREATE TABLE IF NOT EXISTS literature_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ChatMessage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_updated TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES ResearchSession(session_id)
            );

            CREATE TABLE IF NOT EXISTS AssistantMessage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                suggested_topic TEXT,
                created_at REAL NOT NULL
            );
            """
        )
        conn.commit()
        # Lightweight migration for DBs created before acs_components_json existed.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(ResearchSession)")}
        if "acs_components_json" not in existing_cols:
            conn.execute("ALTER TABLE ResearchSession ADD COLUMN acs_components_json TEXT")
            conn.commit()
        if "coherence_json" not in existing_cols:
            conn.execute("ALTER TABLE ResearchSession ADD COLUMN coherence_json TEXT")
            conn.commit()
        if "source" not in existing_cols:
            # Default 'live': anything predating this column is assumed to be
            # real app usage unless proven otherwise (see scripts/backfill_session_source.py
            # for the one-time reclassification of pre-existing seed sessions).
            conn.execute("ALTER TABLE ResearchSession ADD COLUMN source TEXT NOT NULL DEFAULT 'live'")
            conn.commit()
    finally:
        conn.close()
