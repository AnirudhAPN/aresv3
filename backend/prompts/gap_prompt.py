"""Prompt template for the Gap Analysis Agent.

Enforces (in the prompt itself) the rule that gap claims must be evidence-
bounded and never framed as absolute claims about the entire field.
"""
import json

GAP_SCHEMA_HINT = """{
  "current_trends": ["..."],
  "gaps": [
    {"gap_description": "...", "based_on_papers": ["paper title or id", "..."], "confidence_note": "..."}
  ],
  "future_directions": ["..."],
  "low_confidence_gaps": true
}"""


def build_gap_prompt(topic: str, papers: list[dict], low_coverage: bool) -> str:
    papers_block = json.dumps(
        [{"title": p["title"], "summary": p.get("summary") or p.get("abstract", "")} for p in papers],
        indent=2,
    )
    coverage_note = (
        "\nNOTE: literature retrieval flagged LOW COVERAGE (fewer than 3 relevant "
        "papers found). You MUST set low_confidence_gaps=true and explicitly note "
        "in each confidence_note that conclusions are based on limited retrieved "
        "literature.\n"
        if low_coverage
        else ""
    )
    return f"""You are the Gap Analysis Agent in a multi-agent research assistant.

Research topic: "{topic}"

You are given a SMALL SAMPLE of retrieved papers below — not the full field.
{coverage_note}
Papers reviewed:
{papers_block}

STRICT RULES:
- NEVER claim an absolute research gap (e.g. "no one has studied X"). You do
  not have access to the full literature.
- Every claimed gap MUST reference specific paper(s) from the list above by
  title, in `based_on_papers`.
- Use qualified language: "not addressed in the papers reviewed here" rather
  than "no research exists on this."
- `confidence_note` must state how many papers the observation is based on,
  e.g. "based on {len(papers)} retrieved papers."

Produce:
- 3-5 current_trends observed across the reviewed papers.
- 2-4 gaps, each evidence-bounded per the rules above.
- 2-4 future_directions grounded in the reviewed papers.
- low_confidence_gaps: {"true" if low_coverage else "true or false, based on your judgment of evidence strength"}.

Respond with ONLY valid JSON matching this exact shape, no prose:
{GAP_SCHEMA_HINT}
"""
