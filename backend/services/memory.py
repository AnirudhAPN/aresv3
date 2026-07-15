"""
Persistence layer: stores each research session (topic, all five agent
outputs, ACS scores, retry counts, parse failures, flags, timestamp) and
supports history retrieval.
"""
from __future__ import annotations

import json
import time
import uuid
from difflib import SequenceMatcher
from typing import Any, Optional

from database.sqlite import get_connection, init_db
from models.schemas import ResearchState

init_db()

SIMILAR_TOPIC_THRESHOLD = 0.6


class MemoryService:
    def create_session(self, topic: str, user_id: str = "default") -> str:
        session_id = str(uuid.uuid4())
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO ResearchSession
                    (session_id, user_id, topic, status, current_agent,
                     completed_agents_json, acs_scores_json, acs_components_json,
                     retry_counts_json, parse_failures_json, flags_json, created_at)
                VALUES (?, ?, ?, 'running', NULL, '[]', '{}', '{}', '{}', '{}', '[]', ?)
                """,
                (session_id, user_id, topic, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        return session_id

    def update_status(
        self,
        session_id: str,
        status: str,
        current_agent: Optional[str] = None,
        completed_agents: Optional[list[str]] = None,
        acs_scores: Optional[dict[str, float]] = None,
        acs_components: Optional[dict[str, dict[str, float]]] = None,
        retry_counts: Optional[dict[str, int]] = None,
        parse_failures: Optional[dict[str, bool]] = None,
        flags: Optional[list[str]] = None,
    ) -> None:
        conn = get_connection()
        try:
            fields, values = ["status = ?"], [status]
            if current_agent is not None:
                fields.append("current_agent = ?")
                values.append(current_agent)
            if completed_agents is not None:
                fields.append("completed_agents_json = ?")
                values.append(json.dumps(completed_agents))
            if acs_scores is not None:
                fields.append("acs_scores_json = ?")
                values.append(json.dumps(acs_scores))
            if acs_components is not None:
                fields.append("acs_components_json = ?")
                values.append(json.dumps(acs_components))
            if retry_counts is not None:
                fields.append("retry_counts_json = ?")
                values.append(json.dumps(retry_counts))
            if parse_failures is not None:
                fields.append("parse_failures_json = ?")
                values.append(json.dumps(parse_failures))
            if flags is not None:
                fields.append("flags_json = ?")
                values.append(json.dumps(flags))
            values.append(session_id)
            conn.execute(
                f"UPDATE ResearchSession SET {', '.join(fields)} WHERE session_id = ?",
                values,
            )
            conn.commit()
        finally:
            conn.close()

    def save_agent_output(
        self, session_id: str, agent_name: str, output: dict[str, Any] | None, parse_failed: bool
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO AgentOutputs (session_id, agent_name, output_json, parse_failed)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, agent_name, json.dumps(output) if output else None, int(parse_failed)),
            )
            conn.commit()
        finally:
            conn.close()

    def log_metric(
        self,
        session_id: str,
        agent_name: str,
        confidence: float,
        evidence_quality: float,
        task_completion: float,
        acs: float,
        retries: int,
        execution_time: float,
        condition: str = "adaptive",
        topic: str | None = None,
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO Metrics
                    (session_id, agent_name, condition, topic, confidence,
                     evidence_quality, task_completion, acs, retries, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, agent_name, condition, topic, confidence,
                    evidence_quality, task_completion, acs, retries, execution_time,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM ResearchSession WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            session = dict(row)
            outputs = conn.execute(
                "SELECT agent_name, output_json, parse_failed FROM AgentOutputs "
                "WHERE session_id = ? ORDER BY id DESC",
                (session_id,),
            ).fetchall()
            latest_by_agent: dict[str, Any] = {}
            for r in outputs:
                if r["agent_name"] not in latest_by_agent:
                    latest_by_agent[r["agent_name"]] = {
                        "output": json.loads(r["output_json"]) if r["output_json"] else None,
                        "parse_failed": bool(r["parse_failed"]),
                    }
            session["agent_outputs"] = latest_by_agent
            session["completed_agents"] = json.loads(session.pop("completed_agents_json") or "[]")
            session["acs_scores"] = json.loads(session.pop("acs_scores_json") or "{}")
            session["acs_components"] = json.loads(session.pop("acs_components_json") or "{}")
            session["retry_counts"] = json.loads(session.pop("retry_counts_json") or "{}")
            session["parse_failures"] = json.loads(session.pop("parse_failures_json") or "{}")
            session["flags"] = json.loads(session.pop("flags_json") or "[]")
            return session
        finally:
            conn.close()

    def find_similar_sessions(
        self, topic: str, exclude_session_id: Optional[str] = None, threshold: float = SIMILAR_TOPIC_THRESHOLD
    ) -> list[dict[str, Any]]:
        """Fuzzy-match the given topic against every prior session's topic
        (own research history, not published literature — see
        services/similarity.py for the retrieved-paper check). Cheap string
        similarity is enough here since topics are short and this only
        needs to catch near-duplicate phrasing, not semantic paraphrases."""
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT session_id, topic FROM ResearchSession WHERE status = 'completed'"
            ).fetchall()
        finally:
            conn.close()

        matches = []
        normalized_topic = topic.strip().lower()
        for row in rows:
            if row["session_id"] == exclude_session_id:
                continue
            similarity = SequenceMatcher(None, normalized_topic, row["topic"].strip().lower()).ratio()
            if similarity >= threshold:
                matches.append({"session_id": row["session_id"], "topic": row["topic"], "similarity": similarity})
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:5]

    def list_history(self, user_id: str = "default") -> list[dict[str, Any]]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT session_id, topic, status, created_at FROM ResearchSession "
                "ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def add_chat_message(
        self, session_id: str, role: str, content: str, agent_updated: Optional[str] = None
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO ChatMessage (session_id, role, content, agent_updated, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, agent_updated, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_chat_history(self, session_id: str) -> list[dict[str, Any]]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT role, content, agent_updated, created_at FROM ChatMessage "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- AI Assistant (research planning / workflow guidance) ----------------
    # A single continuous conversation per user, not tied to any research
    # session — distinct from ChatMessage above, which refines one already-
    # completed proposal.

    def add_assistant_message(
        self, role: str, content: str, suggested_topic: Optional[str] = None, user_id: str = "default"
    ) -> None:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO AssistantMessage (user_id, role, content, suggested_topic, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, role, content, suggested_topic, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_assistant_history(self, user_id: str = "default", limit: int = 200) -> list[dict[str, Any]]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT role, content, suggested_topic, created_at FROM AssistantMessage "
                "WHERE user_id = ? ORDER BY id ASC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def clear_assistant_history(self, user_id: str = "default") -> None:
        conn = get_connection()
        try:
            conn.execute("DELETE FROM AssistantMessage WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()


memory_service = MemoryService()
