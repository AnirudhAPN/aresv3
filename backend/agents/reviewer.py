"""Agent 5 — Reviewer Agent.

Part A: deterministic rule-checks, implemented in plain code (no LLM call).
Part B: qualitative critique from a SEPARATE judge model (see
services/llm.py `judge()`), explicitly told Part A's results so it can
reference them rather than duplicate the work.
"""
from __future__ import annotations

import json

from config import settings
from models.schemas import (
    GapOutput,
    LiteratureOutput,
    MethodologyOutput,
    PlannerOutput,
    ReviewerOutput,
)
from prompts.reviewer_prompt import build_reviewer_critique_prompt
from services import similarity
from services.llm import llm_service, _strip_code_fences


def run_rule_checks(
    literature_output: LiteratureOutput,
    gap_output: GapOutput,
    methodology_output: MethodologyOutput,
) -> dict[str, bool]:
    """Part A — deterministic, code-only checks. No LLM involved."""
    checks: dict[str, bool] = {}

    checks["gaps_have_evidence"] = bool(gap_output.gaps) and all(
        len(g.based_on_papers) > 0 for g in gap_output.gaps
    )

    methodology_items = methodology_output.datasets + methodology_output.algorithms
    checks["methodology_items_verified_or_flagged"] = all(
        item.verified for item in methodology_items
    ) or bool(methodology_output.unverified_suggestions)

    checks["sufficient_literature_coverage"] = (
        len(literature_output.papers) >= 3 or literature_output.low_coverage
    )

    checks["literature_has_verifiable_urls"] = bool(literature_output.papers) and all(
        bool(p.url) for p in literature_output.papers
    )

    checks["gap_output_fields_non_empty"] = bool(gap_output.current_trends) and bool(
        gap_output.future_directions
    )

    checks["methodology_fields_non_empty"] = bool(
        methodology_output.evaluation_metrics
    ) and bool(methodology_output.implementation_steps)

    return checks


class ReviewerAgent:
    name = "reviewer"

    def run(
        self,
        topic: str,
        planner_output: PlannerOutput,
        literature_output: LiteratureOutput,
        gap_output: GapOutput,
        methodology_output: MethodologyOutput,
    ) -> tuple[ReviewerOutput, bool]:
        rule_check_results = run_rule_checks(literature_output, gap_output, methodology_output)

        full_proposal = {
            "planner": planner_output.model_dump(),
            "literature": literature_output.model_dump(),
            "gap": gap_output.model_dump(),
            "methodology": methodology_output.model_dump(),
        }

        parse_failed = False
        strengths: list[str] = []
        weaknesses: list[str] = []
        missing_components: list[str] = []
        novelty_score = 0.5
        quality_score = 0.5

        if settings.llm_provider == "local":
            from services.local_model import local_model_service
            gap_descriptions = [g.gap_description for g in gap_output.gaps]
            data = local_model_service.run_reviewer(
                topic, planner_output.research_objectives, gap_descriptions, rule_check_results
            )
            strengths = data["strengths"]
            weaknesses = data["weaknesses"]
            missing_components = data["missing_components"]
            quality_score = data["quality_score"]
        else:
            prompt = build_reviewer_critique_prompt(topic, full_proposal, rule_check_results)
            try:
                raw = llm_service.judge(prompt, agent="reviewer")
                cleaned = _strip_code_fences(raw)
                parsed = json.loads(cleaned)
                strengths = parsed.get("strengths", [])
                weaknesses = parsed.get("weaknesses", [])
                missing_components = parsed.get("missing_components", [])
                novelty_score = float(parsed.get("novelty_score", 0.5))
                quality_score = float(parsed.get("quality_score", 0.5))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                parse_failed = True

        # Overrides whatever novelty_score the generator produced (LLM
        # self-report on the API path, a fixed placeholder on the local
        # path) with a real, evidence-grounded measure: how similar is
        # this topic to the papers actually retrieved for it.
        novelty_score = similarity.compute_novelty_score(topic, literature_output.papers)

        caveats = [
            f"Rule check failed: {name}" for name, passed in rule_check_results.items() if not passed
        ]
        if literature_output.low_coverage:
            caveats.append("Literature coverage was low — treat gaps/methodology with caution.")
        if gap_output.low_confidence_gaps:
            caveats.append("Gap analysis is based on limited retrieved literature.")

        output = ReviewerOutput(
            rule_check_results=rule_check_results,
            strengths=strengths,
            weaknesses=weaknesses,
            missing_components=missing_components,
            novelty_score=novelty_score,
            quality_score=quality_score,
            caveats=caveats,
        )
        return output, parse_failed


reviewer_agent = ReviewerAgent()
