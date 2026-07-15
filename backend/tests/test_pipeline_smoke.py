"""
Smoke test: runs the full orchestrator end-to-end with Gemini, the judge
model, Semantic Scholar, and arXiv all mocked, asserting the pipeline
completes and produces a valid ResearchSession without hitting any real
external API.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mock_generate(prompt: str) -> str:
    """Return a schema-appropriate canned JSON response based on cues in
    the prompt, so the same mock can serve every agent."""
    if "Planner Agent" in prompt:
        return json.dumps(
            {
                "research_objectives": ["Understand X in domain Y"],
                "research_questions": ["How does X affect Y?"],
                "keywords": ["machine learning", "healthcare", "diagnosis"],
                "research_scope": "Focused on applied ML methods for diagnostic support.",
            }
        )
    if "summaries" in prompt.lower() or "Literature Agent" in prompt:
        return json.dumps({"summaries": ["Summary A", "Summary B", "Summary C"]})
    if "Gap Analysis Agent" in prompt:
        return json.dumps(
            {
                "current_trends": ["Trend A", "Trend B"],
                "gaps": [
                    {
                        "gap_description": "Not addressed in the papers reviewed here.",
                        "based_on_papers": ["Paper A"],
                        "confidence_note": "based on 3 retrieved papers",
                    }
                ],
                "future_directions": ["Direction A"],
                "low_confidence_gaps": False,
            }
        )
    if "Methodology Agent" in prompt:
        return json.dumps(
            {
                "datasets": [{"name": "UCI Machine Learning Repository", "url": "https://archive.ics.uci.edu/", "description": "d", "verified": True}],
                "algorithms": [{"name": "Random Forest", "url": "https://scikit-learn.org/stable/modules/ensemble.html#forest", "description": "a", "verified": True}],
                "evaluation_metrics": ["accuracy", "F1"],
                "implementation_steps": ["Collect data", "Train model", "Evaluate"],
                "unverified_suggestions": [],
            }
        )
    return "{}"


def _mock_judge(prompt: str) -> str:
    if "internal_consistency" in prompt:
        return json.dumps({"internal_consistency": 0.9, "specificity": 0.85, "completeness": 0.9})
    return json.dumps(
        {
            "strengths": ["Clear objectives"],
            "weaknesses": ["Limited dataset diversity"],
            "missing_components": ["Ethics discussion"],
            "novelty_score": 0.7,
            "quality_score": 0.75,
        }
    )


def _mock_paper_list():
    from models.schemas import Paper

    return [
        Paper(title="Paper A", authors=["A. Author"], year=2023, venue="Venue A",
              url="https://example.com/a", citation_count=10, source="semantic_scholar",
              abstract="Abstract A about machine learning in healthcare."),
        Paper(title="Paper B", authors=["B. Author"], year=2022, venue="Venue B",
              url="https://example.com/b", citation_count=5, source="arxiv",
              abstract="Abstract B about diagnostic ML models."),
        Paper(title="Paper C", authors=["C. Author"], year=2021, venue="Venue C",
              url="https://example.com/c", citation_count=2, source="semantic_scholar",
              abstract="Abstract C about clinical decision support."),
    ]


class _FakeEmbedder:
    """Deterministic bag-of-words-ish embedder so evidence_quality_gap can
    run offline in tests without downloading a real sentence-transformers
    model from huggingface.co."""

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


def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        with patch("config.settings.database_path", db_path):
            # Re-import modules that read settings.database_path at import
            # time is avoided by patching before any DB-touching import.
            # services.memory only auto-creates tables on its FIRST import in
            # this process, so explicitly re-run init_db() against this
            # test's tmp path in case another test file already imported
            # services.memory (and thus initialized a different DB) earlier
            # in the same pytest session.
            from database.sqlite import init_db
            init_db()
            from services.llm import llm_service
            from services.literature_search import literature_search_service
            from services.memory import memory_service
            import orchestrator as orch

            with patch.object(llm_service, "generate", side_effect=_mock_generate), \
                 patch.object(llm_service, "judge", side_effect=_mock_judge), \
                 patch.object(literature_search_service, "search", return_value=(_mock_paper_list(), False)), \
                 patch("services.collaboration._get_embedder", return_value=_FakeEmbedder()), \
                 patch("services.similarity.get_embedder", return_value=_FakeEmbedder()), \
                 patch("services.url_verify.verify_urls_concurrently", return_value={}):

                session_id = memory_service.create_session("AI in Healthcare")
                final_state = orch.run_pipeline(session_id, "AI in Healthcare", condition="adaptive")

                assert final_state["planner_output"] is not None
                assert final_state["literature_output"] is not None
                assert final_state["gap_output"] is not None
                assert final_state["methodology_output"] is not None
                assert final_state["reviewer_output"] is not None
                assert "reviewer" in final_state.get("completed_agents", [])

                session = memory_service.get_session(session_id)
                assert session["status"] == "completed"
                assert session["topic"] == "AI in Healthcare"

    print("Pipeline smoke test passed.")


if __name__ == "__main__":
    test_pipeline_smoke()
