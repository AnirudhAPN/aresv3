"""
Tests for services/assistant.py: the session-independent AI Assistant Agent
that helps with research planning, suggestions, and workflow guidance —
distinct from services/chat.py, which refines one completed proposal.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_assistant_chat_persists_turn_and_returns_suggested_topic():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            from database.sqlite import init_db
            init_db()
            from services.memory import memory_service
            from services.llm import llm_service
            from services import assistant as assistant_service

            def fake_generate(prompt, temperature=0.4):
                assert "ARES Research Assistant" in prompt
                return json.dumps({
                    "reply": "That's broad — want to narrow it to a specific model type or dataset domain?",
                    "suggested_topic": "Explainable AI for chest X-ray diagnosis",
                })

            with patch.object(llm_service, "generate", side_effect=fake_generate):
                result = assistant_service.chat("I want to research AI in healthcare")

            assert "narrow" in result["reply"].lower()
            assert result["suggested_topic"] == "Explainable AI for chest X-ray diagnosis"

            history = memory_service.get_assistant_history()
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[0]["content"] == "I want to research AI in healthcare"
            assert history[1]["role"] == "assistant"
            assert history[1]["suggested_topic"] == "Explainable AI for chest X-ray diagnosis"

    print("test_assistant_chat_persists_turn_and_returns_suggested_topic passed.")


def test_assistant_chat_falls_back_gracefully_on_malformed_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            from database.sqlite import init_db
            init_db()
            from services.memory import memory_service
            from services.llm import llm_service
            from services import assistant as assistant_service

            with patch.object(llm_service, "generate", return_value="not valid json at all"):
                result = assistant_service.chat("How does the scoring work?")

            # Falls back to the raw model text rather than crashing or dropping the turn.
            assert result["reply"] == "not valid json at all"
            assert result["suggested_topic"] is None

            history = memory_service.get_assistant_history()
            assert len(history) == 2

    print("test_assistant_chat_falls_back_gracefully_on_malformed_json passed.")


def test_assistant_history_is_conversation_scoped_not_session_scoped():
    """Unlike ChatMessage (per research session), AssistantMessage has no
    session_id — it's one continuous conversation, so consecutive chat()
    calls should accumulate in the same history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            from database.sqlite import init_db
            init_db()
            from services.memory import memory_service
            from services.llm import llm_service
            from services import assistant as assistant_service

            with patch.object(
                llm_service, "generate",
                return_value=json.dumps({"reply": "Sure, happy to help.", "suggested_topic": None}),
            ):
                assistant_service.chat("First question")
                assistant_service.chat("Second question")

            history = memory_service.get_assistant_history()
            assert len(history) == 4
            assert [m["content"] for m in history if m["role"] == "user"] == ["First question", "Second question"]

            memory_service.clear_assistant_history()
            assert memory_service.get_assistant_history() == []

    print("test_assistant_history_is_conversation_scoped_not_session_scoped passed.")


if __name__ == "__main__":
    test_assistant_chat_persists_turn_and_returns_suggested_topic()
    test_assistant_chat_falls_back_gracefully_on_malformed_json()
    test_assistant_history_is_conversation_scoped_not_session_scoped()
    print("All assistant tests passed.")
