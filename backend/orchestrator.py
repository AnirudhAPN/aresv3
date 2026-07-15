"""
LangGraph orchestrator for the ARES pipeline.

Each agent is an independent graph node. Structural (parse-failure) retries
are handled INSIDE each agent (see agents/base.py) and are transparent to
the graph. ACS-driven QUALITY retries are handled by the graph itself: a
node computes its ACS score, and a conditional edge (built from the pure
routing functions in services/collaboration.py) decides whether to loop
back into the same node once (with an augmented prompt referencing the
weakest scoring dimension) or advance to the next agent.

The routing decision itself is delegated to `next_node_after` in
services/collaboration.py, which is a pure function of state — this keeps
the graph wiring below purely structural, and means alternate collaboration
strategies (static / adaptive / RL-based) can be swapped in by changing
which routing function is imported, without touching this file's topology.
"""
from __future__ import annotations

import time
from typing import Literal

from langgraph.graph import StateGraph, END

from config import settings
from models.schemas import (
    GapOutput,
    LiteratureOutput,
    MethodologyOutput,
    PlannerOutput,
    ReviewerOutput,
    ResearchState,
)
from agents.planner import planner_agent
from agents.literature import literature_agent
from agents.gap import gap_agent
from agents.methodology import methodology_agent
from agents.reviewer import reviewer_agent
from prompts.planner_prompt import build_planner_prompt
from prompts.gap_prompt import build_gap_prompt
from prompts.methodology_prompt import build_methodology_prompt
from utils.prompt_helpers import build_retry_prompt
from services import collaboration
from services.memory import memory_service

CollaborationCondition = Literal["adaptive", "static", "single_agent"]


def _weakest_dimension(confidence: float, evidence_quality: float, task_completion: float) -> str:
    dims = {
        "confidence": confidence,
        "evidence_quality": evidence_quality,
        "task_completion": task_completion,
    }
    return min(dims, key=dims.get)


def _record_score(
    state: ResearchState,
    agent_name: str,
    confidence: float,
    evidence_quality: float,
    task_completion: float,
    acs: float,
    execution_time: float,
    session_id: str,
    topic: str,
    condition: str,
) -> None:
    state.setdefault("acs_scores", {})[agent_name] = acs
    state.setdefault("acs_components", {})[agent_name] = {
        "confidence": confidence,
        "evidence_quality": evidence_quality,
        "task_completion": task_completion,
    }
    retries = state.get("retry_counts", {}).get(agent_name, 0)
    memory_service.log_metric(
        session_id=session_id,
        agent_name=agent_name,
        confidence=confidence,
        evidence_quality=evidence_quality,
        task_completion=task_completion,
        acs=acs,
        retries=retries,
        execution_time=execution_time,
        condition=condition,
        topic=topic,
    )


def build_graph(condition: CollaborationCondition = "adaptive"):
    """Build the LangGraph StateGraph. `condition` selects the collaboration
    strategy used for evaluation: 'adaptive' (full ACS + retry), 'static'
    (fixed sequence, ACS computed for logging only, never triggers retry),
    or the pipeline is bypassed entirely for 'single_agent' (see
    run_single_agent_baseline below).
    """
    graph = StateGraph(ResearchState)

    retries_enabled = condition == "adaptive"

    # -- Planner node ---------------------------------------------------

    def planner_node(state: ResearchState) -> ResearchState:
        session_id = state["session_id"]
        topic = state["topic"]
        retries_used = state.get("retry_counts", {}).get("planner", 0)
        scratch = state.setdefault("_scratch", {})

        start = time.time()
        if retries_used == 0:
            prompt = build_planner_prompt(topic)
        else:
            prev = scratch.get("planner_prev_output", "")
            weak = scratch.get("planner_weak_dim", "task_completion")
            prompt = build_retry_prompt(build_planner_prompt(topic), prev, weak)

        result = planner_agent.run_with_retry_prompt(prompt) if retries_used else planner_agent.run(topic)
        exec_time = time.time() - start

        state["planner_output"] = result.output
        state.setdefault("parse_failures", {})["planner"] = result.parse_failed
        memory_service.save_agent_output(session_id, "planner", result.output.model_dump(), result.parse_failed)

        output_dict = result.output.model_dump()
        confidence = collaboration.compute_confidence(output_dict)
        evidence_quality = 1.0 if not result.parse_failed else 0.3  # planner has no external evidence source
        task_completion = collaboration.compute_task_completion(output_dict)
        acs = collaboration.compute_acs(confidence, evidence_quality, task_completion)

        _record_score(state, "planner", confidence, evidence_quality, task_completion, acs, exec_time, session_id, topic, condition)
        scratch["planner_prev_output"] = str(output_dict)
        scratch["planner_weak_dim"] = _weakest_dimension(confidence, evidence_quality, task_completion)

        if retries_enabled and collaboration.should_retry_agent(state, "planner"):
            state.setdefault("retry_counts", {})["planner"] = retries_used + 1
        else:
            state.setdefault("completed_agents", []).append("planner")
        memory_service.update_status(
            session_id, status="running", current_agent="planner",
            completed_agents=state.get("completed_agents", []), acs_scores=state.get("acs_scores", {}),
            acs_components=state.get("acs_components", {}),
        )
        return state

    def planner_router(state: ResearchState) -> str:
        if retries_enabled and collaboration.should_retry_agent(state, "planner") and "planner" not in state.get("completed_agents", []):
            return "planner"
        if state.get("acs_scores", {}).get("planner", 1.0) < settings.acs_retry_threshold:
            state.setdefault("flags", []).append("planner_low_confidence")
        return "literature"

    # -- Literature node --------------------------------------------------

    def literature_node(state: ResearchState) -> ResearchState:
        session_id = state["session_id"]
        topic = state["topic"]
        keywords = state["planner_output"].keywords if state.get("planner_output") else [topic]

        start = time.time()
        output, parse_failed = literature_agent.run(topic, keywords)
        exec_time = time.time() - start

        state["literature_output"] = output
        state.setdefault("parse_failures", {})["literature"] = parse_failed
        memory_service.save_agent_output(session_id, "literature", output.model_dump(), parse_failed)

        output_dict = output.model_dump()
        confidence = collaboration.compute_confidence(output_dict)
        evidence_quality = collaboration.evidence_quality_literature(output)
        task_completion = collaboration.compute_task_completion(output_dict)
        acs = collaboration.compute_acs(confidence, evidence_quality, task_completion)

        _record_score(state, "literature", confidence, evidence_quality, task_completion, acs, exec_time, session_id, topic, condition)

        if output.low_coverage:
            state.setdefault("flags", []).append("literature_low_coverage")

        retries_used = state.get("retry_counts", {}).get("literature", 0)
        if retries_enabled and acs < settings.acs_retry_threshold and retries_used < settings.max_acs_retries_per_agent:
            # Literature retry re-runs synthesis/retrieval once; retrieval
            # itself is deterministic given keywords, so this mainly gives
            # the synthesis step another attempt.
            state.setdefault("retry_counts", {})["literature"] = retries_used + 1
        else:
            state.setdefault("completed_agents", []).append("literature")
        memory_service.update_status(
            session_id, status="running", current_agent="literature",
            completed_agents=state.get("completed_agents", []), acs_scores=state.get("acs_scores", {}),
            acs_components=state.get("acs_components", {}),
        )
        return state

    def literature_router(state: ResearchState) -> str:
        retries_used = state.get("retry_counts", {}).get("literature", 0)
        if (
            retries_enabled
            and state.get("acs_scores", {}).get("literature", 1.0) < settings.acs_retry_threshold
            and retries_used <= settings.max_acs_retries_per_agent
            and "literature" not in state.get("completed_agents", [])
        ):
            return "literature"
        return "gap"

    # -- Gap node -----------------------------------------------------------

    def gap_node(state: ResearchState) -> ResearchState:
        session_id = state["session_id"]
        topic = state["topic"]
        lit_output: LiteratureOutput = state["literature_output"]
        retries_used = state.get("retry_counts", {}).get("gap", 0)
        scratch = state.setdefault("_scratch", {})

        start = time.time()
        if retries_used == 0:
            result = gap_agent.run(topic, lit_output)
        else:
            base_prompt = build_gap_prompt(topic, [p.model_dump() for p in lit_output.papers], lit_output.low_coverage)
            prev = scratch.get("gap_prev_output", "")
            weak = scratch.get("gap_weak_dim", "evidence_quality")
            retry_prompt = build_retry_prompt(base_prompt, prev, weak)
            result = gap_agent.generate_validated(retry_prompt, GapOutput)
            if lit_output.low_coverage:
                result.output.low_confidence_gaps = True
        exec_time = time.time() - start

        state["gap_output"] = result.output
        state.setdefault("parse_failures", {})["gap"] = result.parse_failed
        memory_service.save_agent_output(session_id, "gap", result.output.model_dump(), result.parse_failed)

        output_dict = result.output.model_dump()
        confidence = collaboration.compute_confidence(output_dict)
        evidence_quality = collaboration.evidence_quality_gap(result.output, lit_output)
        task_completion = collaboration.compute_task_completion(output_dict)
        acs = collaboration.compute_acs(confidence, evidence_quality, task_completion)

        _record_score(state, "gap", confidence, evidence_quality, task_completion, acs, exec_time, session_id, topic, condition)
        scratch["gap_prev_output"] = str(output_dict)
        scratch["gap_weak_dim"] = _weakest_dimension(confidence, evidence_quality, task_completion)

        if result.output.low_confidence_gaps:
            state.setdefault("flags", []).append("gap_low_confidence")

        if retries_enabled and collaboration.should_retry_agent(state, "gap"):
            state.setdefault("retry_counts", {})["gap"] = retries_used + 1
        else:
            state.setdefault("completed_agents", []).append("gap")
        memory_service.update_status(
            session_id, status="running", current_agent="gap",
            completed_agents=state.get("completed_agents", []), acs_scores=state.get("acs_scores", {}),
            acs_components=state.get("acs_components", {}),
        )
        return state

    def gap_router(state: ResearchState) -> str:
        if retries_enabled and collaboration.should_retry_agent(state, "gap") and "gap" not in state.get("completed_agents", []):
            return "gap"
        return "methodology"

    # -- Methodology node -----------------------------------------------------

    def methodology_node(state: ResearchState) -> ResearchState:
        session_id = state["session_id"]
        topic = state["topic"]
        planner_output: PlannerOutput = state["planner_output"]
        retries_used = state.get("retry_counts", {}).get("methodology", 0)
        scratch = state.setdefault("_scratch", {})

        start = time.time()
        if retries_used == 0:
            result = methodology_agent.run(topic, planner_output.research_questions)
        else:
            base_prompt = build_methodology_prompt(
                topic,
                planner_output.research_questions,
                methodology_agent.known_datasets,
                methodology_agent.known_algorithms,
            )
            prev = scratch.get("methodology_prev_output", "")
            weak = scratch.get("methodology_weak_dim", "evidence_quality")
            retry_prompt = build_retry_prompt(base_prompt, prev, weak)
            result = methodology_agent.generate_validated(retry_prompt, MethodologyOutput)
        exec_time = time.time() - start

        state["methodology_output"] = result.output
        state.setdefault("parse_failures", {})["methodology"] = result.parse_failed
        memory_service.save_agent_output(session_id, "methodology", result.output.model_dump(), result.parse_failed)

        output_dict = result.output.model_dump()
        confidence = collaboration.compute_confidence(output_dict)
        evidence_quality = collaboration.evidence_quality_methodology(result.output)
        task_completion = collaboration.compute_task_completion(output_dict)
        acs = collaboration.compute_acs(confidence, evidence_quality, task_completion)

        _record_score(state, "methodology", confidence, evidence_quality, task_completion, acs, exec_time, session_id, topic, condition)
        scratch["methodology_prev_output"] = str(output_dict)
        scratch["methodology_weak_dim"] = _weakest_dimension(confidence, evidence_quality, task_completion)

        if retries_enabled and collaboration.should_retry_agent(state, "methodology"):
            state.setdefault("retry_counts", {})["methodology"] = retries_used + 1
        else:
            state.setdefault("completed_agents", []).append("methodology")
        memory_service.update_status(
            session_id, status="running", current_agent="methodology",
            completed_agents=state.get("completed_agents", []), acs_scores=state.get("acs_scores", {}),
            acs_components=state.get("acs_components", {}),
        )
        return state

    def methodology_router(state: ResearchState) -> str:
        if retries_enabled and collaboration.should_retry_agent(state, "methodology") and "methodology" not in state.get("completed_agents", []):
            return "methodology"
        return "reviewer"

    # -- Reviewer node --------------------------------------------------------

    def reviewer_node(state: ResearchState) -> ResearchState:
        session_id = state["session_id"]
        topic = state["topic"]

        start = time.time()
        output, parse_failed = reviewer_agent.run(
            topic,
            state["planner_output"],
            state["literature_output"],
            state["gap_output"],
            state["methodology_output"],
        )
        exec_time = time.time() - start

        state["reviewer_output"] = output
        state.setdefault("parse_failures", {})["reviewer"] = parse_failed
        memory_service.save_agent_output(session_id, "reviewer", output.model_dump(), parse_failed)

        output_dict = output.model_dump()
        confidence = collaboration.compute_confidence(output_dict)
        evidence_quality = collaboration.evidence_quality_reviewer(output_dict, output.rule_check_results)
        task_completion = collaboration.compute_task_completion(output_dict)
        acs = collaboration.compute_acs(confidence, evidence_quality, task_completion)

        _record_score(state, "reviewer", confidence, evidence_quality, task_completion, acs, exec_time, session_id, topic, condition)

        if any(not passed for passed in output.rule_check_results.values()):
            state.setdefault("flags", []).append("reviewer_rule_check_failed")

        state.setdefault("completed_agents", []).append("reviewer")
        return state

    graph.add_node("planner", planner_node)
    graph.add_node("literature", literature_node)
    graph.add_node("gap", gap_node)
    graph.add_node("methodology", methodology_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", planner_router, {"planner": "planner", "literature": "literature"})
    graph.add_conditional_edges("literature", literature_router, {"literature": "literature", "gap": "gap"})
    graph.add_conditional_edges("gap", gap_router, {"gap": "gap", "methodology": "methodology"})
    graph.add_conditional_edges("methodology", methodology_router, {"methodology": "methodology", "reviewer": "reviewer"})
    graph.add_edge("reviewer", END)

    return graph.compile()


def run_pipeline(session_id: str, topic: str, condition: CollaborationCondition = "adaptive") -> ResearchState:
    """Synchronous entry point used by the FastAPI BackgroundTask. Builds and
    runs the graph, updating session status/polling fields in SQLite as
    agents complete."""
    app_graph = build_graph(condition)

    initial_state: ResearchState = {
        "session_id": session_id,
        "topic": topic,
        "acs_scores": {},
        "acs_components": {},
        "retry_counts": {},
        "structural_retry_counts": {},
        "parse_failures": {},
        "flags": [],
        "completed_agents": [],
    }

    try:
        final_state = app_graph.invoke(initial_state, config={"recursion_limit": 50})
        memory_service.update_status(
            session_id,
            status="completed",
            current_agent=None,
            completed_agents=final_state.get("completed_agents", []),
            acs_scores=final_state.get("acs_scores", {}),
            acs_components=final_state.get("acs_components", {}),
            retry_counts=final_state.get("retry_counts", {}),
            parse_failures=final_state.get("parse_failures", {}),
            flags=final_state.get("flags", []),
        )
        return final_state
    except Exception as exc:  # noqa: BLE001 — must not crash the background task
        memory_service.update_status(session_id, status="failed", flags=[f"error: {exc}"])
        raise


def run_single_agent_baseline(session_id: str, topic: str) -> None:
    """Evaluation condition (1): one Gemini call producing the whole
    proposal directly, using the same output schema, for comparison against
    the multi-agent conditions."""
    import json
    from services.llm import llm_service, _strip_code_fences

    schema_hint = """{
  "planner": {"research_objectives": [], "research_questions": [], "keywords": [], "research_scope": ""},
  "literature": {"papers": [], "low_coverage": true, "keywords_used": []},
  "gap": {"current_trends": [], "gaps": [], "future_directions": [], "low_confidence_gaps": true},
  "methodology": {"datasets": [], "algorithms": [], "evaluation_metrics": [], "implementation_steps": [], "unverified_suggestions": []},
  "reviewer": {"rule_check_results": {}, "strengths": [], "weaknesses": [], "missing_components": [], "novelty_score": 0.0, "quality_score": 0.0, "caveats": []}
}"""
    prompt = f"""Produce a complete research proposal for the topic "{topic}" in a single
pass, covering planning, literature synthesis, gap analysis, methodology,
and self-review. NOTE: as a single-pass baseline, you do NOT have access to
real retrieved papers — you may describe representative literature themes,
but do not fabricate specific citations with fake URLs.

Respond with ONLY valid JSON matching this shape:
{schema_hint}
"""
    start = time.time()
    try:
        raw = llm_service.generate(prompt)
        parsed = json.loads(_strip_code_fences(raw))
    except Exception:
        parsed = {}
    exec_time = time.time() - start
    memory_service.log_metric(
        session_id=session_id,
        agent_name="single_agent_baseline",
        confidence=0.0,
        evidence_quality=0.0,
        task_completion=1.0 if parsed else 0.0,
        acs=0.0,
        retries=0,
        execution_time=exec_time,
        condition="single_agent",
        topic=topic,
    )
    memory_service.save_agent_output(session_id, "single_agent_baseline", parsed, not bool(parsed))
