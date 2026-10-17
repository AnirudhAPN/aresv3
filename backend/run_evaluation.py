"""
ARES Evaluation Runner
======================
Runs the ARES pipeline across 10 fixed research domains and 3 collaboration
conditions (single_agent, static, adaptive), logging all metrics to the
Metrics table in SQLite.

Usage:
    cd backend
    python run_evaluation.py

The script runs topics SEQUENTIALLY (SQLite serializes writes; Gemini/Anthropic
have rate limits) with configurable delays between runs. Results are exported
to evaluation_results.csv and a summary is printed to stdout.

Expected runtime: ~30-60 minutes depending on API rate limits.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

# Ensure we can import from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.sqlite import get_connection, init_db
from orchestrator import run_pipeline, run_single_agent_baseline
from services.memory import memory_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── The 10 fixed evaluation topics (from README) ──────────────────────────
EVAL_TOPICS = [
    "AI in Agriculture",
    "Cybersecurity",
    "Healthcare",
    "Robotics",
    "Education",
    "Climate",
    "Blockchain",
    "IoT",
    "Forensics",
    "Finance",
]

# Conditions to evaluate
CONDITIONS = ["single_agent", "static", "adaptive"]

# Delay between runs to respect API rate limits (seconds)
DELAY_BETWEEN_RUNS = 30


def run_single_topic(topic: str, condition: str) -> dict:
    """Run one topic under one condition. Returns a summary dict."""
    session_id = memory_service.create_session(topic)
    logger.info("  Session %s started for '%s' [%s]", session_id, topic, condition)

    start = time.time()
    success = True
    error_msg = None

    try:
        if condition == "single_agent":
            run_single_agent_baseline(session_id, topic)
            memory_service.update_status(session_id, status="completed")
        else:
            run_pipeline(session_id, topic, condition)
    except Exception as exc:
        success = False
        error_msg = str(exc)[:200]
        logger.error("  FAILED: %s", error_msg)

    elapsed = time.time() - start
    logger.info("  Completed in %.1fs (success=%s)", elapsed, success)

    return {
        "session_id": session_id,
        "topic": topic,
        "condition": condition,
        "success": success,
        "error": error_msg,
        "total_time": round(elapsed, 2),
    }


def extract_metrics() -> list[dict]:
    """Pull all metrics from the Metrics table."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT m.session_id, m.agent_name, m.condition, m.topic,
                   m.confidence, m.evidence_quality, m.task_completion,
                   m.acs, m.retries, m.execution_time
            FROM Metrics m
            ORDER BY m.id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def extract_session_details(session_ids: list[str]) -> list[dict]:
    """Pull coherence, flags, and retry counts for completed sessions."""
    results = []
    for sid in session_ids:
        session = memory_service.get_session(sid)
        if session:
            results.append({
                "session_id": sid,
                "topic": session.get("topic", ""),
                "status": session.get("status", ""),
                "coherence": session.get("coherence"),
                "flags": session.get("flags", []),
                "retry_counts": session.get("retry_counts", {}),
                "parse_failures": session.get("parse_failures", {}),
                "acs_scores": session.get("acs_scores", {}),
                "acs_components": session.get("acs_components", {}),
            })
    return results


def export_csv(metrics: list[dict], filename: str = "evaluation_results.csv"):
    """Export metrics to CSV."""
    if not metrics:
        logger.warning("No metrics to export.")
        return
    fieldnames = list(metrics[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    logger.info("Exported %d metric rows to %s", len(metrics), filename)


def print_summary(run_results: list[dict], metrics: list[dict], sessions: list[dict]):
    """Print a human-readable summary of the evaluation."""
    print("\n" + "=" * 80)
    print("ARES EVALUATION SUMMARY")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Success rate by condition
    print("\n── Run Success Rate ──")
    for cond in CONDITIONS:
        runs = [r for r in run_results if r["condition"] == cond]
        successes = sum(1 for r in runs if r["success"])
        print(f"  {cond:15s}: {successes}/{len(runs)} succeeded")

    # Average ACS by condition and agent
    print("\n── Average ACS by Condition × Agent ──")
    print(f"  {'Agent':<15s}", end="")
    for cond in CONDITIONS:
        print(f"  {cond:>12s}", end="")
    print()
    print("  " + "-" * 55)

    agents = ["planner", "literature", "gap", "methodology", "reviewer"]
    for agent in agents:
        print(f"  {agent:<15s}", end="")
        for cond in CONDITIONS:
            scores = [m["acs"] for m in metrics if m["agent_name"] == agent and m["condition"] == cond and m["acs"] is not None]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {avg:>12.3f}", end="")
        print()

    # ACS component breakdown for adaptive condition
    print("\n── ACS Component Breakdown (Adaptive) ──")
    print(f"  {'Agent':<15s}  {'Confidence':>12s}  {'Evidence':>12s}  {'Completion':>12s}  {'ACS':>12s}")
    print("  " + "-" * 67)
    for agent in agents:
        agent_metrics = [m for m in metrics if m["agent_name"] == agent and m["condition"] == "adaptive"]
        if not agent_metrics:
            continue
        avg_c = sum(m["confidence"] or 0 for m in agent_metrics) / len(agent_metrics)
        avg_e = sum(m["evidence_quality"] or 0 for m in agent_metrics) / len(agent_metrics)
        avg_t = sum(m["task_completion"] or 0 for m in agent_metrics) / len(agent_metrics)
        avg_acs = sum(m["acs"] or 0 for m in agent_metrics) / len(agent_metrics)
        print(f"  {agent:<15s}  {avg_c:>12.3f}  {avg_e:>12.3f}  {avg_t:>12.3f}  {avg_acs:>12.3f}")

    # Execution time
    print("\n── Average Execution Time per Agent (Adaptive, seconds) ──")
    for agent in agents:
        times = [m["execution_time"] for m in metrics if m["agent_name"] == agent and m["condition"] == "adaptive" and m["execution_time"] is not None]
        avg_t = sum(times) / len(times) if times else 0
        print(f"  {agent:<15s}: {avg_t:>8.1f}s")

    # Coherence scores (adaptive only)
    print("\n── Cross-Agent Coherence (Adaptive) ──")
    coherence_scores = [s["coherence"]["overall_score"] for s in sessions if s.get("coherence")]
    if coherence_scores:
        print(f"  Mean overall coherence: {sum(coherence_scores)/len(coherence_scores):.3f}")
        print(f"  Min:  {min(coherence_scores):.3f}")
        print(f"  Max:  {max(coherence_scores):.3f}")
    else:
        print("  No coherence data available.")

    # Retry and failure summary
    print("\n── Retry & Parse Failure Summary (Adaptive) ──")
    total_retries = 0
    total_parse_fails = 0
    for s in sessions:
        for agent, count in s.get("retry_counts", {}).items():
            total_retries += count
        for agent, failed in s.get("parse_failures", {}).items():
            if failed:
                total_parse_fails += 1
    print(f"  Total ACS retries triggered: {total_retries}")
    print(f"  Total parse failures:        {total_parse_fails}")

    # Flags
    print("\n── Flags Raised (Adaptive) ──")
    all_flags = {}
    for s in sessions:
        for flag in s.get("flags", []):
            all_flags[flag] = all_flags.get(flag, 0) + 1
    if all_flags:
        for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            print(f"  {flag}: {count}")
    else:
        print("  No flags raised.")

    print("\n" + "=" * 80)


def main():
    init_db()
    logger.info("=" * 60)
    logger.info("ARES Evaluation — %d topics × %d conditions = %d runs",
                len(EVAL_TOPICS), len(CONDITIONS), len(EVAL_TOPICS) * len(CONDITIONS))
    logger.info("=" * 60)

    run_results = []
    all_session_ids = []

    for i, topic in enumerate(EVAL_TOPICS):
        for j, condition in enumerate(CONDITIONS):
            run_idx = i * len(CONDITIONS) + j + 1
            total = len(EVAL_TOPICS) * len(CONDITIONS)
            logger.info("[%d/%d] Topic: '%s' | Condition: %s", run_idx, total, topic, condition)

            result = run_single_topic(topic, condition)
            run_results.append(result)
            all_session_ids.append(result["session_id"])

            # Delay between runs to avoid rate limiting
            if run_idx < total:
                logger.info("  Waiting %ds before next run...", DELAY_BETWEEN_RUNS)
                time.sleep(DELAY_BETWEEN_RUNS)

    # Extract and export results
    logger.info("Extracting metrics from database...")
    metrics = extract_metrics()
    export_csv(metrics)

    logger.info("Extracting session details...")
    sessions = extract_session_details(all_session_ids)

    # Export session details as JSON
    with open("evaluation_sessions.json", "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, default=str)
    logger.info("Exported session details to evaluation_sessions.json")

    # Print summary
    print_summary(run_results, metrics, sessions)

    # Save run results
    with open("evaluation_runs.json", "w", encoding="utf-8") as f:
        json.dump(run_results, f, indent=2)
    logger.info("Exported run results to evaluation_runs.json")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
