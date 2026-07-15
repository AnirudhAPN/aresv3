"""
Adaptive Collaboration Score (ACS) computation and routing logic.

ACS = confidence_weight * confidence
    + evidence_weight * evidence_quality
    + completion_weight * task_completion

All three dimensions are computed independently:
  - confidence: a SEPARATE lightweight LLM-as-judge call (see services/llm.py
    `judge()`), given only the agent's output, never the generating call
    itself. This avoids circular self-reported confidence.
  - evidence_quality: programmatic, agent-specific (see below).
  - task_completion: rule-based structural completeness check.

All conditional-edge functions used by the LangGraph orchestrator are pure
functions of `ResearchState`, defined here (not as inline lambdas in
orchestrator.py) so they're independently testable and swappable.
"""
from __future__ import annotations

import json
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from models.schemas import (
    GapOutput,
    LiteratureOutput,
    MethodologyOutput,
    ResearchState,
    ReviewerOutput,
)
from services.embeddings import get_embedder as _get_embedder
from services.llm import llm_service


# ---------------------------------------------------------------------------
# CONFIDENCE — separate LLM-as-judge call
# ---------------------------------------------------------------------------

CONFIDENCE_JUDGE_PROMPT = """You are scoring the quality of an AI agent's output in \
isolation. You are NOT told what prompt produced it. Rate the following JSON output \
on three criteria, each 0.0-1.0:

1. internal_consistency: does the output contradict itself anywhere?
2. specificity: are claims concrete and specific, or generic filler?
3. completeness: are all fields present and non-trivially filled?

Agent output to evaluate:
{output_json}

Respond with ONLY a JSON object of this exact shape, no other text:
{{"internal_consistency": <float>, "specificity": <float>, "completeness": <float>}}
"""


def compute_confidence(agent_output: dict[str, Any]) -> float:
    """Separate LLM-as-judge call scoring the output in isolation."""
    prompt = CONFIDENCE_JUDGE_PROMPT.format(output_json=json.dumps(agent_output, indent=2))
    try:
        result = llm_service.judge_json(prompt)
        scores = [
            float(result.get("internal_consistency", 0.5)),
            float(result.get("specificity", 0.5)),
            float(result.get("completeness", 0.5)),
        ]
        return sum(scores) / len(scores)
    except Exception:
        # Judge call failed to parse/execute — fall back to a neutral score
        # rather than crashing the pipeline.
        return 0.5


# ---------------------------------------------------------------------------
# EVIDENCE QUALITY — programmatic, agent-specific
# ---------------------------------------------------------------------------

def evidence_quality_literature(output: LiteratureOutput) -> float:
    if not output.papers:
        return 0.0
    # Prefer the real HTTP-verified signal when available (url_verify has
    # run); fall back to "url is present" for papers that haven't been
    # checked yet so this doesn't regress to 0 before verification completes.
    checked = [p for p in output.papers if p.url_verified is not None]
    if checked:
        resolvable = sum(1 for p in checked if p.url_verified)
        base = resolvable / len(checked)
    else:
        resolvable = sum(1 for p in output.papers if p.url)
        base = resolvable / len(output.papers)
    if output.low_coverage:
        base *= len(output.papers) / max(settings.min_papers_for_full_coverage, 1)
    return max(0.0, min(1.0, base))


def evidence_quality_gap(gap_output: GapOutput, literature_output: LiteratureOutput) -> float:
    if not gap_output.gaps or not literature_output.papers:
        return 0.0
    embedder = _get_embedder()
    lit_texts = [f"{p.title}. {p.summary or p.abstract}" for p in literature_output.papers]
    lit_embeddings = embedder.encode(lit_texts)

    scores = []
    for gap in gap_output.gaps:
        gap_embedding = embedder.encode([gap.gap_description])
        sims = cosine_similarity(gap_embedding, lit_embeddings)[0]
        scores.append(float(max(sims)))
    return sum(scores) / len(scores) if scores else 0.0


def evidence_quality_methodology(output: MethodologyOutput) -> float:
    items = output.datasets + output.algorithms
    if not items:
        return 0.0
    # Whitelist match (verified) is the primary signal; for items that also
    # carry a URL that's been HTTP-checked, a dead link pulls the score down
    # even if the name matched the whitelist.
    scores = []
    for item in items:
        score = 1.0 if item.verified else 0.0
        if item.url_verified is False:
            score = min(score, 0.3)
        scores.append(score)
    return sum(scores) / len(scores)


def evidence_quality_reviewer(reviewer_raw: dict[str, Any], rule_check_results: dict[str, bool]) -> float:
    """Agreement rate between Part A (rule checks) and Part B (LLM critique).

    Approximated as: if rule checks mostly pass, does the critique lean
    positive (few weaknesses relative to strengths), and vice versa.
    """
    if not rule_check_results:
        return 0.5
    rule_pass_rate = sum(1 for v in rule_check_results.values() if v) / len(rule_check_results)

    strengths = len(reviewer_raw.get("strengths", []))
    weaknesses = len(reviewer_raw.get("weaknesses", []))
    total_points = strengths + weaknesses
    critique_positivity = strengths / total_points if total_points else 0.5

    # Agreement = how close the two independent signals are.
    return 1.0 - abs(rule_pass_rate - critique_positivity)


# ---------------------------------------------------------------------------
# TASK COMPLETION — rule-based structural completeness
# ---------------------------------------------------------------------------

def _non_empty_fraction(model_dict: dict[str, Any]) -> float:
    if not model_dict:
        return 0.0
    filled = 0
    for value in model_dict.values():
        if isinstance(value, (list, dict, str)):
            if len(value) > 0:
                filled += 1
        elif value is not None:
            filled += 1
    return filled / len(model_dict)


def compute_task_completion(agent_output: dict[str, Any]) -> float:
    return _non_empty_fraction(agent_output)


# ---------------------------------------------------------------------------
# ACS aggregation
# ---------------------------------------------------------------------------

def compute_acs(confidence: float, evidence_quality: float, task_completion: float) -> float:
    w = settings.acs_weights
    return (
        w.confidence_weight * confidence
        + w.evidence_weight * evidence_quality
        + w.completion_weight * task_completion
    )


# ---------------------------------------------------------------------------
# Pure routing functions used as LangGraph conditional edges
# ---------------------------------------------------------------------------

def should_retry_agent(state: ResearchState, agent_name: str) -> bool:
    """True if the agent's most recent ACS score is below threshold and it
    hasn't already used its single allotted retry."""
    acs = state.get("acs_scores", {}).get(agent_name)
    retries_used = state.get("retry_counts", {}).get(agent_name, 0)
    if acs is None:
        return False
    return acs < settings.acs_retry_threshold and retries_used < settings.max_acs_retries_per_agent


def should_structural_retry(state: ResearchState, agent_name: str) -> bool:
    retries_used = state.get("structural_retry_counts", {}).get(agent_name, 0)
    return (
        state.get("parse_failures", {}).get(agent_name, False)
        and retries_used < settings.max_structural_retries_per_agent
    )


def next_node_after(state: ResearchState, agent_name: str, next_agent: str) -> str:
    """Static/sequential collaboration strategy: proceed once retries (both
    structural and ACS-based) are exhausted or unnecessary. This function
    signature is intentionally simple so alternative collaboration
    strategies (adaptive skip-ahead, RL-based routing) can be swapped in by
    implementing the same `(state, agent_name, next_agent) -> str` contract.
    """
    if should_structural_retry(state, agent_name):
        return agent_name  # loop back to same agent for structural retry
    if should_retry_agent(state, agent_name):
        return agent_name  # loop back to same agent for ACS-driven retry
    return next_agent
