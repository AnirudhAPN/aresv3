"""Schema validation tests for each agent's Pydantic output model."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import (
    DatasetRec,
    GapOutput,
    LiteratureOutput,
    MethodologyOutput,
    Paper,
    PlannerOutput,
    ResearchGap,
    ReviewerOutput,
)


def test_planner_output_defaults():
    p = PlannerOutput()
    assert p.research_objectives == []
    assert p.research_scope == ""


def test_planner_output_valid():
    p = PlannerOutput(
        research_objectives=["Investigate X"],
        research_questions=["How does X affect Y?"],
        keywords=["x", "y"],
        research_scope="Focused on X in domain Y.",
    )
    assert len(p.research_objectives) == 1


def test_paper_requires_source_and_url():
    with pytest.raises(ValidationError):
        Paper(title="T", url="https://example.com")  # missing `source`

    p = Paper(title="T", url="https://example.com", source="arxiv")
    assert p.verified is True


def test_paper_rejects_invalid_source():
    with pytest.raises(ValidationError):
        Paper(title="T", url="https://example.com", source="google")


def test_literature_output_low_coverage_flag():
    out = LiteratureOutput(papers=[], low_coverage=True, keywords_used=["x"])
    assert out.low_coverage is True
    assert out.papers == []


def test_research_gap_requires_description():
    with pytest.raises(ValidationError):
        ResearchGap(based_on_papers=["Paper A"])  # missing gap_description

    gap = ResearchGap(gap_description="Not addressed", based_on_papers=["Paper A"])
    assert gap.based_on_papers == ["Paper A"]


def test_gap_output_defaults():
    out = GapOutput()
    assert out.low_confidence_gaps is False
    assert out.gaps == []


def test_dataset_rec_verified_defaults_false():
    d = DatasetRec(name="Some Dataset")
    assert d.verified is False


def test_methodology_output_structure():
    out = MethodologyOutput(
        datasets=[DatasetRec(name="UCI ML Repository", url="https://archive.ics.uci.edu/", verified=True)],
        algorithms=[DatasetRec(name="Random Forest", verified=True)],
        evaluation_metrics=["accuracy", "F1"],
        implementation_steps=["Step 1", "Step 2"],
        unverified_suggestions=[],
    )
    assert out.datasets[0].verified is True


def test_reviewer_output_score_types():
    out = ReviewerOutput(
        rule_check_results={"gaps_have_evidence": True},
        strengths=["Clear scope"],
        weaknesses=["Thin methodology"],
        missing_components=[],
        novelty_score=0.7,
        quality_score=0.6,
        caveats=[],
    )
    assert isinstance(out.novelty_score, float)
    assert 0.0 <= out.novelty_score <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
