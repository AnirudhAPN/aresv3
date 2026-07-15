"""
Conversational refinement layer for a completed research session.

After the initial five-agent pipeline finishes, the user can keep sending
follow-up messages. Each message is classified as either:

  - a QUESTION about the existing proposal, answered directly from the
    stored session context (no agent is re-run, nothing is mutated), or
  - a REFINEMENT request targeting one specific agent's section, which
    re-runs that agent with the user's feedback folded into its original
    prompt, re-validates and re-scores it via ACS exactly as the
    orchestrator would, and persists the new output.

Because the Reviewer's rule-checks and critique depend on every other
agent's output, refining any non-Reviewer agent automatically triggers a
Reviewer refresh afterward so the report stays internally consistent
without requiring a full pipeline re-run.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from agents.base import BaseAgent
from agents.literature import literature_agent
from agents.methodology import methodology_agent
from agents.reviewer import reviewer_agent
from models.schemas import (
    GapOutput,
    LiteratureOutput,
    MethodologyOutput,
    PlannerOutput,
)
from prompts.gap_prompt import build_gap_prompt
from prompts.methodology_prompt import build_methodology_prompt
from prompts.planner_prompt import build_planner_prompt
from services import collaboration
from services.kaggle_links import resolve_kaggle_url
from services.llm import llm_service, _strip_code_fences
from services.memory import memory_service
from utils.text import strip_leading_ordinal

REFINABLE_AGENTS = ["planner", "literature", "gap", "methodology", "reviewer"]

_SCHEMA_BY_AGENT = {
    "planner": PlannerOutput,
    "gap": GapOutput,
    "methodology": MethodologyOutput,
}

_base_agent = BaseAgent()


INTENT_PROMPT = """You are routing a follow-up message in an ongoing conversation about an
AI-generated research proposal. The proposal has five sections, produced by
five agents: planner, literature, gap, methodology, reviewer.

Current proposal (condensed):
{proposal_summary}

User's follow-up message: "{message}"

Decide whether the user is:
(a) asking a QUESTION about the existing proposal (wants an explanation,
    clarification, or opinion — nothing needs to change), or
(b) requesting a CHANGE to one specific section (e.g. "add more datasets",
    "focus the gap analysis on X", "narrow the research questions").

If (b), identify which single section is the best fit: planner, literature,
gap, methodology, or reviewer.

Respond with ONLY valid JSON, no prose:
{{"target": "question" | "planner" | "literature" | "gap" | "methodology" | "reviewer",
  "instruction": "<if target is not 'question', a clean one-sentence restatement of what the user wants changed, otherwise empty string>"}}
"""

QUESTION_PROMPT = """You are ARES, an AI research assistant. Answer the user's question about
the research proposal below, using only information present in the proposal. If the
answer isn't in the proposal, say so plainly rather than inventing details.

Research topic: {topic}

Proposal:
{proposal_json}

User's question: "{message}"

Answer conversationally in 2-5 sentences. Do not restate the whole proposal.
"""


def _condensed_summary(session: dict[str, Any]) -> str:
    outputs = session.get("agent_outputs", {})
    parts = []
    for agent in REFINABLE_AGENTS:
        entry = outputs.get(agent) or {}
        output = entry.get("output") or {}
        parts.append(f"- {agent}: {json.dumps(output)[:400]}")
    return "\n".join(parts)


def _classify(message: str, session: dict[str, Any]) -> dict[str, str]:
    prompt = INTENT_PROMPT.format(proposal_summary=_condensed_summary(session), message=message)
    try:
        raw = llm_service.generate(prompt, temperature=0.1)
        parsed = json.loads(_strip_code_fences(raw))
        target = parsed.get("target", "question")
        if target not in ("question", *REFINABLE_AGENTS):
            target = "question"
        return {"target": target, "instruction": parsed.get("instruction", message)}
    except Exception:
        # If routing fails for any reason, default to treating it as a
        # question rather than silently mutating a section.
        return {"target": "question", "instruction": ""}


def _answer_question(message: str, session: dict[str, Any]) -> str:
    outputs = {
        agent: (session.get("agent_outputs", {}).get(agent) or {}).get("output")
        for agent in REFINABLE_AGENTS
    }
    prompt = QUESTION_PROMPT.format(
        topic=session.get("topic", ""),
        proposal_json=json.dumps(outputs, indent=2)[:6000],
        message=message,
    )
    try:
        return llm_service.generate(prompt, temperature=0.4).strip()
    except Exception:
        return (
            "I couldn't reach the generating model to answer that just now. "
            "Please check the backend logs and try again."
        )


def _score_and_persist(
    session_id: str, agent_name: str, output_dict: dict[str, Any], evidence_quality: float
) -> float:
    confidence = collaboration.compute_confidence(output_dict)
    task_completion = collaboration.compute_task_completion(output_dict)
    acs = collaboration.compute_acs(confidence, evidence_quality, task_completion)

    session = memory_service.get_session(session_id)
    acs_scores = {**session.get("acs_scores", {}), agent_name: acs}
    acs_components = {
        **session.get("acs_components", {}),
        agent_name: {
            "confidence": confidence,
            "evidence_quality": evidence_quality,
            "task_completion": task_completion,
        },
    }
    memory_service.update_status(
        session_id,
        status="completed",
        acs_scores=acs_scores,
        acs_components=acs_components,
    )
    return acs


def _refine_planner(session_id: str, topic: str, current: dict, instruction: str) -> PlannerOutput:
    base_prompt = build_planner_prompt(topic)
    prompt = (
        f"{base_prompt}\n\n--- REFINEMENT REQUEST ---\nCurrent output:\n{json.dumps(current)}\n\n"
        f"The user asked: \"{instruction}\"\n\nRevise the output to address this request while "
        f"still satisfying all original requirements. Respond with ONLY valid JSON in the same schema."
    )
    result = _base_agent.generate_validated(prompt, PlannerOutput)
    memory_service.save_agent_output(session_id, "planner", result.output.model_dump(), result.parse_failed)
    return result.output


def _refine_literature(session_id: str, topic: str, current: dict, instruction: str) -> LiteratureOutput:
    # Literature is retrieval-first: refine by deriving adjusted search
    # keywords from the feedback, then re-running real retrieval + synthesis
    # rather than asking the LLM to edit paper metadata directly.
    keyword_prompt = f"""Current search keywords: {json.dumps(current.get("keywords_used", []))}
Topic: {topic}
User feedback on the literature review: "{instruction}"

Propose an updated list of 6-10 search keywords that would better satisfy this
feedback when querying Semantic Scholar/arXiv. Respond with ONLY valid JSON:
{{"keywords": ["...", "..."]}}
"""
    try:
        raw = llm_service.generate(keyword_prompt, temperature=0.3)
        new_keywords = json.loads(_strip_code_fences(raw)).get("keywords") or current.get("keywords_used", [])
    except Exception:
        new_keywords = current.get("keywords_used", [])

    output, parse_failed = literature_agent.run(topic, new_keywords)
    memory_service.save_agent_output(session_id, "literature", output.model_dump(), parse_failed)
    return output


def _refine_gap(session_id: str, topic: str, current: dict, instruction: str, literature: dict) -> GapOutput:
    papers = literature.get("papers", [])
    base_prompt = build_gap_prompt(topic, papers, literature.get("low_coverage", False))
    prompt = (
        f"{base_prompt}\n\n--- REFINEMENT REQUEST ---\nCurrent output:\n{json.dumps(current)}\n\n"
        f"The user asked: \"{instruction}\"\n\nRevise the output to address this request while "
        f"still following all original rules (evidence-bounded claims, based_on_papers required). "
        f"Respond with ONLY valid JSON in the same schema."
    )
    result = _base_agent.generate_validated(prompt, GapOutput)
    if literature.get("low_coverage"):
        result.output.low_confidence_gaps = True
    memory_service.save_agent_output(session_id, "gap", result.output.model_dump(), result.parse_failed)
    return result.output


def _refine_methodology(session_id: str, topic: str, current: dict, instruction: str, research_questions: list[str]) -> MethodologyOutput:
    base_prompt = build_methodology_prompt(
        topic, research_questions, methodology_agent.known_datasets, methodology_agent.known_algorithms
    )
    prompt = (
        f"{base_prompt}\n\n--- REFINEMENT REQUEST ---\nCurrent output:\n{json.dumps(current)}\n\n"
        f"The user asked: \"{instruction}\"\n\nRevise the output to address this request while still "
        f"selecting from the whitelist first. Respond with ONLY valid JSON in the same schema."
    )
    result = _base_agent.generate_validated(prompt, MethodologyOutput)
    if isinstance(result.output, MethodologyOutput):
        dataset_names = {d["name"].lower() for d in methodology_agent.known_datasets}
        algorithm_names = {a["name"].lower() for a in methodology_agent.known_algorithms}
        for rec in result.output.datasets:
            rec.verified = rec.name.lower() in dataset_names
            rec.kaggle_url = resolve_kaggle_url(rec.name, methodology_agent.known_datasets)
        for rec in result.output.algorithms:
            rec.verified = rec.name.lower() in algorithm_names
        result.output.implementation_steps = [
            strip_leading_ordinal(s) for s in result.output.implementation_steps
        ]
    memory_service.save_agent_output(session_id, "methodology", result.output.model_dump(), result.parse_failed)
    return result.output


def _refresh_reviewer(session_id: str, topic: str, session: dict[str, Any]) -> None:
    outputs = session["agent_outputs"]
    planner = PlannerOutput(**(outputs.get("planner", {}).get("output") or {}))
    literature = LiteratureOutput(**(outputs.get("literature", {}).get("output") or {}))
    gap = GapOutput(**(outputs.get("gap", {}).get("output") or {}))
    methodology = MethodologyOutput(**(outputs.get("methodology", {}).get("output") or {}))

    output, parse_failed = reviewer_agent.run(topic, planner, literature, gap, methodology)
    memory_service.save_agent_output(session_id, "reviewer", output.model_dump(), parse_failed)

    output_dict = output.model_dump()
    evidence_quality = collaboration.evidence_quality_reviewer(output_dict, output.rule_check_results)
    _score_and_persist(session_id, "reviewer", output_dict, evidence_quality)


def handle_chat_message(session_id: str, message: str) -> dict[str, Any]:
    """Main entry point. Returns {"reply": str, "agent_updated": str | None,
    "session": dict} after processing one follow-up message."""
    session = memory_service.get_session(session_id)
    if session is None:
        raise ValueError("session not found")
    if session.get("status") != "completed":
        return {
            "reply": "This research session is still running — please wait for it to complete before refining it.",
            "agent_updated": None,
            "session": session,
        }

    topic = session["topic"]
    memory_service.add_chat_message(session_id, "user", message)

    intent = _classify(message, session)
    target = intent["target"]
    instruction = intent["instruction"] or message

    if target == "question":
        reply = _answer_question(message, session)
        memory_service.add_chat_message(session_id, "assistant", reply, agent_updated=None)
        session = memory_service.get_session(session_id)
        return {"reply": reply, "agent_updated": None, "session": session}

    outputs = session["agent_outputs"]

    if target == "planner":
        current = outputs.get("planner", {}).get("output") or {}
        new_output = _refine_planner(session_id, topic, current, instruction)
        output_dict = new_output.model_dump()
        _score_and_persist(session_id, "planner", output_dict, evidence_quality=1.0 if output_dict else 0.3)

    elif target == "literature":
        current = outputs.get("literature", {}).get("output") or {}
        new_output = _refine_literature(session_id, topic, current, instruction)
        output_dict = new_output.model_dump()
        eq = collaboration.evidence_quality_literature(new_output)
        _score_and_persist(session_id, "literature", output_dict, eq)

    elif target == "gap":
        current = outputs.get("gap", {}).get("output") or {}
        literature = outputs.get("literature", {}).get("output") or {}
        new_output = _refine_gap(session_id, topic, current, instruction, literature)
        output_dict = new_output.model_dump()
        lit_model = LiteratureOutput(**literature) if literature else LiteratureOutput(papers=[])
        eq = collaboration.evidence_quality_gap(new_output, lit_model)
        _score_and_persist(session_id, "gap", output_dict, eq)

    elif target == "methodology":
        current = outputs.get("methodology", {}).get("output") or {}
        planner_output = outputs.get("planner", {}).get("output") or {}
        research_questions = planner_output.get("research_questions", [])
        new_output = _refine_methodology(session_id, topic, current, instruction, research_questions)
        output_dict = new_output.model_dump()
        eq = collaboration.evidence_quality_methodology(new_output)
        _score_and_persist(session_id, "methodology", output_dict, eq)

    elif target == "reviewer":
        session = memory_service.get_session(session_id)
        _refresh_reviewer(session_id, topic, session)

    else:
        target = "question"
        reply = _answer_question(message, session)
        memory_service.add_chat_message(session_id, "assistant", reply, agent_updated=None)
        session = memory_service.get_session(session_id)
        return {"reply": reply, "agent_updated": None, "session": session}

    # Reviewer depends on every other agent's output, so keep it in sync
    # whenever a non-Reviewer section changes.
    if target != "reviewer":
        session = memory_service.get_session(session_id)
        _refresh_reviewer(session_id, topic, session)

    session = memory_service.get_session(session_id)
    reply = f"Updated the {target} section based on your feedback"
    if target != "reviewer":
        reply += " (and refreshed the reviewer's feedback to stay consistent)."
    else:
        reply += "."
    memory_service.add_chat_message(session_id, "assistant", reply, agent_updated=target)

    return {"reply": reply, "agent_updated": target, "session": session}
