"""
Simple SQLite-backed TTL cache for literature retrieval results, keyed by a
normalized keyword-set. Avoids redundant Semantic Scholar / arXiv calls and
keeps repeated runs of the same topic within rate limits.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

from config import settings


class LiteratureCache:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.database_path
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS literature_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def make_key(keywords: list[str]) -> str:
        normalized = sorted(k.strip().lower() for k in keywords if k.strip())
        return "|".join(normalized)

    def get(self, keywords: list[str]) -> Optional[dict]:
        key = self.make_key(keywords)
        ttl_seconds = settings.literature_cache_ttl_days * 86400
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, created_at FROM literature_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        payload_json, created_at = row
        if time.time() - created_at > ttl_seconds:
            return None  # expired
        return json.loads(payload_json)

    def set(self, keywords: list[str], payload: dict) -> None:
        key = self.make_key(keywords)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO literature_cache (cache_key, payload_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (key, json.dumps(payload), time.time()),
            )
            conn.commit()


literature_cache = LiteratureCache()
