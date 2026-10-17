"""Builds a LoRA fine-tuning dataset from real ARES pipeline runs, using
TASK-DECOMPOSED (prompt, completion) pairs — not whole-agent-object pairs.

Why decomposed: asking a 77M-parameter model to emit one whole nested JSON
object (4-6 fields, some nested) in a single generation is an unfair task
for that model size and collapsed to invalid JSON in earlier rounds even as
training loss dropped. Each agent's real output is instead sliced into
several small, single-purpose (prompt, completion) pairs — "list 3-4
research objectives" -> a short JSON array, "summarize this one paper" ->
one sentence — matching the SAME decomposition services/local_model.py uses
at inference time (both import the shared prompt builders in
services/local_decomposed.py, so training and inference prompts are always
shaped identically).

Fields that are constructed deterministically at inference time (dataset/
algorithm whitelist selection, gap evidence citations, reviewer novelty/
quality scores — see services/local_model.py) are NOT included here; there
is nothing useful to fine-tune a generator on for those.

As more sessions are generated, re-run this script to grow the dataset —
it is fully idempotent and reads only from the DB.

Output: backend/data/finetune/raw_pairs.jsonl
  {"agent": ..., "session_id": ..., "prompt": ..., "completion": ...}

Usage:
    python scripts/build_finetune_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.sqlite import get_connection
from services import local_decomposed as ld

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"


def _load_agent_outputs(conn, session_id: str) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT agent_name, output_json, parse_failed FROM AgentOutputs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    outputs: dict[str, dict] = {}
    for row in rows:
        if row["parse_failed"] or not row["output_json"]:
            continue
        try:
            outputs[row["agent_name"]] = json.loads(row["output_json"])
        except json.JSONDecodeError:
            continue
    return outputs


def _list_pair(agent: str, session_id: str, prompt: str, items: list[str]) -> dict | None:
    if not items:
        return None
    return {
        "agent": agent,
        "session_id": session_id,
        "prompt": prompt,
        "completion": json.dumps(items, ensure_ascii=False),
    }


def build_pairs() -> list[dict]:
    conn = get_connection()
    pairs: list[dict] = []
    try:
        # Only 'seed' sessions (curated, mocked-generation training data) are
        # ever fine-tuned on. Real app usage ('live', the default source) is
        # excluded so ordinary usage can never silently feed back into the
        # next training round and corrupt it — see services/memory.py.
        sessions = conn.execute(
            "SELECT session_id, topic, status FROM ResearchSession "
            "WHERE status = 'completed' AND source = 'seed'"
        ).fetchall()

        for session in sessions:
            session_id, topic = session["session_id"], session["topic"]
            outputs = _load_agent_outputs(conn, session_id)

            planner = outputs.get("planner")
            literature = outputs.get("literature")
            gap = outputs.get("gap")
            methodology = outputs.get("methodology")
            reviewer = outputs.get("reviewer")

            # -- Planner: 4 small pairs instead of 1 big one -----------------
            if planner:
                p = _list_pair(
                    "planner", session_id,
                    ld.planner_objectives_prompt(topic), planner.get("research_objectives", []),
                )
                if p:
                    pairs.append(p)
                p = _list_pair(
                    "planner", session_id,
                    ld.planner_questions_prompt(topic), planner.get("research_questions", []),
                )
                if p:
                    pairs.append(p)
                p = _list_pair(
                    "planner", session_id,
                    ld.planner_keywords_prompt(topic), planner.get("keywords", []),
                )
                if p:
                    pairs.append(p)
                scope = planner.get("research_scope", "")
                if scope:
                    pairs.append({
                        "agent": "planner",
                        "session_id": session_id,
                        "prompt": ld.planner_scope_prompt(topic),
                        "completion": scope,
                    })

            # -- Literature: one pair PER PAPER, not one pair for the batch --
            if literature:
                papers = literature.get("papers", [])
                for paper in papers:
                    summary = paper.get("summary", "")
                    if not summary:
                        continue
                    pairs.append({
                        "agent": "literature",
                        "session_id": session_id,
                        "prompt": ld.literature_summary_prompt(
                            topic, paper.get("title", ""), paper.get("abstract", "")
                        ),
                        "completion": summary,
                    })

            # -- Gap: 3 small pairs; gap descriptions trained WITHOUT the
            # constant prefix (added in code at inference), citations/notes
            # are never generated so never trained on. -----------------------
            if gap and literature:
                papers_for_gap = literature.get("papers", [])
                p = _list_pair(
                    "gap", session_id,
                    ld.gap_trends_prompt(topic, papers_for_gap), gap.get("current_trends", []),
                )
                if p:
                    pairs.append(p)

                bare_descriptions = [
                    g.get("gap_description", "").removeprefix(ld.GAP_DESCRIPTION_PREFIX)
                    for g in gap.get("gaps", [])
                    if g.get("gap_description")
                ]
                p = _list_pair(
                    "gap", session_id,
                    ld.gap_descriptions_prompt(topic, papers_for_gap), bare_descriptions,
                )
                if p:
                    pairs.append(p)

                p = _list_pair(
                    "gap", session_id,
                    ld.gap_future_directions_prompt(topic), gap.get("future_directions", []),
                )
                if p:
                    pairs.append(p)

            # -- Methodology: 3 small pairs; datasets/algorithms are picked
            # deterministically at inference, so never trained on. -----------
            if methodology:
                p = _list_pair(
                    "methodology", session_id,
                    ld.methodology_metrics_prompt(topic), methodology.get("evaluation_metrics", []),
                )
                if p:
                    pairs.append(p)
                p = _list_pair(
                    "methodology", session_id,
                    ld.methodology_steps_prompt(topic), methodology.get("implementation_steps", []),
                )
                if p:
                    pairs.append(p)
                p = _list_pair(
                    "methodology", session_id,
                    ld.methodology_unverified_prompt(topic), methodology.get("unverified_suggestions", []),
                )
                if p:
                    pairs.append(p)

            # -- Reviewer: 3 small pairs; novelty/quality scores are derived
            # from rule-check results at inference, so never trained on. -----
            if reviewer and planner and gap:
                objectives = planner.get("research_objectives", [])
                gap_descriptions = [g.get("gap_description", "") for g in gap.get("gaps", [])]
                p = _list_pair(
                    "reviewer", session_id,
                    ld.reviewer_strengths_prompt(topic, objectives, gap_descriptions),
                    reviewer.get("strengths", []),
                )
                if p:
                    pairs.append(p)
                p = _list_pair(
                    "reviewer", session_id,
                    ld.reviewer_weaknesses_prompt(topic, objectives, gap_descriptions),
                    reviewer.get("weaknesses", []),
                )
                if p:
                    pairs.append(p)
                p = _list_pair(
                    "reviewer", session_id,
                    ld.reviewer_missing_prompt(topic, objectives, gap_descriptions),
                    reviewer.get("missing_components", []),
                )
                if p:
                    pairs.append(p)
    finally:
        conn.close()
    return pairs


def write_outputs(pairs: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "raw_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main() -> None:
    pairs = build_pairs()
    write_outputs(pairs)

    by_agent: dict[str, int] = {}
    for p in pairs:
        by_agent[p["agent"]] = by_agent.get(p["agent"], 0) + 1

    print(f"Wrote {len(pairs)} decomposed training pairs to {OUT_DIR}")
    for agent, count in sorted(by_agent.items()):
        print(f"  {agent}: {count}")


if __name__ == "__main__":
    main()
