"""
Tests for services/chat.py: the follow-up conversation layer that lets a
user keep refining a completed research session after the initial pipeline
run, or ask questions about it without mutating anything.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mock_paper_list():
    from models.schemas import Paper

    return [
        Paper(title="Paper A", authors=["A. Author"], year=2023, venue="Venue A",
              url="https://example.com/a", citation_count=10, source="semantic_scholar",
              abstract="Abstract A about federated learning security."),
        Paper(title="Paper B", authors=["B. Author"], year=2022, venue="Venue B",
              url="https://example.com/b", citation_count=5, source="arxiv",
              abstract="Abstract B about IoT intrusion detection."),
        Paper(title="Paper C", authors=["C. Author"], year=2021, venue="Venue C",
              url="https://example.com/c", citation_count=2, source="semantic_scholar",
              abstract="Abstract C about robust aggregation."),
    ]


class _FakeEmbedder:
    def encode(self, texts):
        import numpy as np

        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(64)
            for w in words:
                vec[hash(w) % 64] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        return np.array(vectors)


def _seed_completed_session(memory_service, session_id: str, topic: str) -> None:
    """Directly populate a completed session's agent outputs, bypassing the
    full orchestrator run, so chat tests can focus on the refinement layer."""
    planner_out = {
        "research_objectives": ["Investigate FL robustness"],
        "research_questions": ["How does poisoning affect FedAvg convergence?"],
        "keywords": ["federated learning", "security"],
        "research_scope": "Cross-device federated learning for IoT security.",
    }
    literature_out = {
        "papers": [p.model_dump() for p in _mock_paper_list()],
        "low_coverage": False,
        "keywords_used": ["federated learning", "security"],
    }
    gap_out = {
        "current_trends": ["Growing interest in robust aggregation"],
        "gaps": [{"gap_description": "Not addressed here: adaptive poisoning under IoT churn.",
                   "based_on_papers": ["Paper A"], "confidence_note": "based on 3 retrieved papers"}],
        "future_directions": ["Adaptive client selection"],
        "low_confidence_gaps": False,
    }
    methodology_out = {
        "datasets": [{"name": "CIC-IoT2023", "url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
                       "description": "d", "verified": True}],
        "algorithms": [{"name": "Federated Averaging (FedAvg)", "url": "https://arxiv.org/abs/1602.05629",
                          "description": "a", "verified": True}],
        "evaluation_metrics": ["accuracy"],
        "implementation_steps": ["Step 1", "Step 2"],
        "unverified_suggestions": [],
    }
    reviewer_out = {
        "rule_check_results": {"gaps_have_evidence": True},
        "strengths": ["Clear scope"], "weaknesses": ["Thin methodology"],
        "missing_components": [], "novelty_score": 0.7, "quality_score": 0.6, "caveats": [],
    }

    for agent, output in [
        ("planner", planner_out), ("literature", literature_out), ("gap", gap_out),
        ("methodology", methodology_out), ("reviewer", reviewer_out),
    ]:
        memory_service.save_agent_output(session_id, agent, output, parse_failed=False)

    memory_service.update_status(
        session_id, status="completed", current_agent=None,
        completed_agents=["planner", "literature", "gap", "methodology", "reviewer"],
        acs_scores={"planner": 0.9, "literature": 0.8, "gap": 0.75, "methodology": 0.85, "reviewer": 0.8},
        acs_components={
            "planner": {"confidence": 0.9, "evidence_quality": 1.0, "task_completion": 0.9},
            "literature": {"confidence": 0.8, "evidence_quality": 0.8, "task_completion": 0.8},
            "gap": {"confidence": 0.7, "evidence_quality": 0.75, "task_completion": 0.8},
            "methodology": {"confidence": 0.85, "evidence_quality": 1.0, "task_completion": 0.8},
            "reviewer": {"confidence": 0.8, "evidence_quality": 0.8, "task_completion": 0.8},
        },
    )


def test_chat_question_does_not_mutate_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            from database.sqlite import init_db
            init_db()
            from services.memory import memory_service
            from services.llm import llm_service
            from services import chat as chat_service

            session_id = memory_service.create_session("Federated Learning for IoT Security")
            _seed_completed_session(memory_service, session_id, "Federated Learning for IoT Security")
            before = memory_service.get_session(session_id)

            def fake_generate(prompt, temperature=0.4):
                if "routing a follow-up" in prompt:
                    return json.dumps({"target": "question", "instruction": ""})
                return "The literature review covers three papers on federated learning and IoT security."

            with patch.object(llm_service, "generate", side_effect=fake_generate):
                result = chat_service.handle_chat_message(session_id, "What does the literature review cover?")

            assert result["agent_updated"] is None
            assert "federated learning" in result["reply"].lower()

            after = memory_service.get_session(session_id)
            assert after["agent_outputs"]["literature"]["output"] == before["agent_outputs"]["literature"]["output"]
            assert after["acs_scores"] == before["acs_scores"]

            history = memory_service.get_chat_history(session_id)
            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "assistant"

    print("test_chat_question_does_not_mutate_session passed.")


def test_chat_refinement_updates_target_agent_and_cascades_reviewer():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            from database.sqlite import init_db
            init_db()
            from services.memory import memory_service
            from services.llm import llm_service
            from services import chat as chat_service
            import services.collaboration as collaboration

            session_id = memory_service.create_session("Federated Learning for IoT Security")
            _seed_completed_session(memory_service, session_id, "Federated Learning for IoT Security")

            def fake_generate(prompt, temperature=0.4):
                if "routing a follow-up" in prompt:
                    return json.dumps({"target": "gap", "instruction": "Add a gap about privacy leakage."})
                if "REFINEMENT REQUEST" in prompt and "gap_description" in prompt or "gaps" in prompt:
                    return json.dumps({
                        "current_trends": ["Growing interest in robust aggregation"],
                        "gaps": [
                            {"gap_description": "Not addressed here: adaptive poisoning under IoT churn.",
                             "based_on_papers": ["Paper A"], "confidence_note": "based on 3 retrieved papers"},
                            {"gap_description": "Not addressed here: privacy leakage via gradient inversion.",
                             "based_on_papers": ["Paper B"], "confidence_note": "based on 3 retrieved papers"},
                        ],
                        "future_directions": ["Adaptive client selection", "Differential privacy for gradients"],
                        "low_confidence_gaps": False,
                    })
                # Reviewer Part B critique call (uses judge(), not generate() — but
                # in case it falls through here, return a safe default)
                return json.dumps({"strengths": [], "weaknesses": [], "missing_components": [],
                                    "novelty_score": 0.7, "quality_score": 0.65})

            def fake_judge(prompt, temperature=0.0):
                return json.dumps({
                    "strengths": ["Now covers privacy leakage explicitly"],
                    "weaknesses": ["Still limited dataset diversity"],
                    "missing_components": [],
                    "novelty_score": 0.75, "quality_score": 0.7,
                })

            with patch.object(llm_service, "generate", side_effect=fake_generate), \
                 patch.object(llm_service, "judge", side_effect=fake_judge), \
                 patch.object(collaboration, "_get_embedder", return_value=_FakeEmbedder()):
                result = chat_service.handle_chat_message(
                    session_id, "Can you add a gap about privacy leakage risks?"
                )

            assert result["agent_updated"] == "gap"
            session = result["session"]
            gap_output = session["agent_outputs"]["gap"]["output"]
            assert len(gap_output["gaps"]) == 2
            assert any("privacy leakage" in g["gap_description"] for g in gap_output["gaps"])

            # Reviewer should have been refreshed as a side effect since gap changed.
            reviewer_output = session["agent_outputs"]["reviewer"]["output"]
            assert reviewer_output["novelty_score"] == 0.75

            history = memory_service.get_chat_history(session_id)
            assert history[-1]["agent_updated"] == "gap"

    print("test_chat_refinement_updates_target_agent_and_cascades_reviewer passed.")


def test_chat_on_running_session_is_rejected_gracefully():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            from database.sqlite import init_db
            init_db()
            from services.memory import memory_service
            from services import chat as chat_service

            session_id = memory_service.create_session("Some Topic")
            # Leave status as "running" (default from create_session).

            result = chat_service.handle_chat_message(session_id, "Add more datasets please.")
            assert result["agent_updated"] is None
            assert "still running" in result["reply"].lower()

    print("test_chat_on_running_session_is_rejected_gracefully passed.")


if __name__ == "__main__":
    test_chat_question_does_not_mutate_session()
    test_chat_refinement_updates_target_agent_and_cascades_reviewer()
    test_chat_on_running_session_is_rejected_gracefully()
    print("All chat tests passed.")
