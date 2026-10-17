"""Task-decomposed prompts for the local fine-tuned models.

The API path (Gemini/Claude) can take one giant prompt and reliably return
a whole nested JSON object — that's a fair ask for a 100B+ parameter model.
It is NOT a fair ask for a 77M-parameter flan-t5-small fine-tuned on a
few dozen examples. Asking it to jointly get 4-6 fields AND nested objects
AND correct JSON syntax right in one shot is why earlier rounds kept
collapsing to invalid JSON even as training loss dropped.

The fix: break each agent's output into several small, single-purpose
generations (topic -> short JSON list of strings, or topic -> one
sentence), and construct anything whitelist-grounded or purely mechanical
(dataset/algorithm selection, gap evidence citations, novelty/quality
scores) in code instead of asking the model to invent it.

These prompt builders are shared between:
  - scripts/build_finetune_dataset.py  (produces training pairs)
  - services/local_model.py            (produces inference-time prompts)
so the model always sees, at inference time, prompts shaped exactly like
what it trained on.
"""
from __future__ import annotations

import json
import re


def _list_prompt(topic: str, instruction: str, context: str = "") -> str:
    ctx = f"\n{context}" if context else ""
    return f"Research topic: {topic}{ctx}\nTask: {instruction}\nRespond with ONLY a JSON array of strings."


def _text_prompt(topic: str, instruction: str, context: str = "") -> str:
    ctx = f"\n{context}" if context else ""
    return f"Research topic: {topic}{ctx}\nTask: {instruction}\nRespond with ONLY the requested text, no JSON, no extra commentary."


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def planner_objectives_prompt(topic: str) -> str:
    return _list_prompt(topic, "List 3 to 4 concrete research objectives for this topic.")


def planner_questions_prompt(topic: str) -> str:
    return _list_prompt(topic, "List 3 to 4 specific research questions for this topic.")


def planner_keywords_prompt(topic: str) -> str:
    return _list_prompt(topic, "List 4 to 5 short keyword phrases for searching literature on this topic.")


def planner_scope_prompt(topic: str) -> str:
    return _text_prompt(topic, "Write a one-paragraph research scope statement describing what this research includes and excludes.")


PLANNER_FIELDS = ["objectives", "questions", "keywords", "scope"]


# ---------------------------------------------------------------------------
# Literature — one generation per retrieved paper (not one call for all)
# ---------------------------------------------------------------------------

def literature_summary_prompt(topic: str, paper_title: str, paper_abstract: str) -> str:
    abstract = (paper_abstract or "")[:400]
    return (
        f"Research topic: {topic}\n"
        f"Paper title: {paper_title}\n"
        f"Abstract: {abstract}\n"
        f"Task: Write ONE sentence describing this paper's relevance to the research topic.\n"
        f"Respond with ONLY the sentence, no JSON, no extra commentary."
    )


# ---------------------------------------------------------------------------
# Gap — trends / bare gap descriptions (prefix added in code) / directions
# ---------------------------------------------------------------------------

def _titles_context(papers: list[dict]) -> str:
    titles = [p.get("title", "") for p in papers[:8] if p.get("title")]
    return f"Retrieved paper titles: {'; '.join(titles)}" if titles else ""


def gap_trends_prompt(topic: str, papers: list[dict]) -> str:
    return _list_prompt(
        topic, "List 3 to 4 current trends in this research area based on these papers.",
        _titles_context(papers),
    )


def gap_descriptions_prompt(topic: str, papers: list[dict]) -> str:
    return _list_prompt(
        topic,
        "List 2 to 3 specific gaps in the literature — things NOT addressed by these papers. "
        "Describe each gap in one plain sentence, with NO prefix phrase.",
        _titles_context(papers),
    )


def gap_future_directions_prompt(topic: str) -> str:
    return _list_prompt(topic, "List 3 future research directions for this topic.")


GAP_DESCRIPTION_PREFIX = "Not addressed in the papers reviewed here: "


# ---------------------------------------------------------------------------
# Methodology — datasets/algorithms picked deterministically (see
# local_model.py); only these three fields are model-generated.
# ---------------------------------------------------------------------------

def methodology_metrics_prompt(topic: str) -> str:
    return _list_prompt(topic, "List 4 evaluation metrics appropriate for this research.")


def methodology_steps_prompt(topic: str) -> str:
    return _list_prompt(
        topic,
        "List 5 to 6 concrete ordered implementation steps for this research. "
        "Do not number them yourself.",
    )


def methodology_unverified_prompt(topic: str) -> str:
    return _list_prompt(
        topic,
        "List 2 to 3 relevant dataset or tool names that would need manual verification "
        "before use (do not repeat well-known standard benchmarks).",
    )


# ---------------------------------------------------------------------------
# Reviewer — strengths/weaknesses/missing only; scores are deterministic
# (see local_model.py) since a small model reliably outputting a
# calibrated float is a much harder ask than reliably outputting a list.
# ---------------------------------------------------------------------------

def _proposal_context(objectives: list[str], gap_descriptions: list[str]) -> str:
    obj = "; ".join(objectives[:3])
    gaps = "; ".join(gap_descriptions[:2])
    parts = []
    if obj:
        parts.append(f"Research objectives: {obj}")
    if gaps:
        parts.append(f"Identified gaps: {gaps}")
    return "\n".join(parts)


def reviewer_strengths_prompt(topic: str, objectives: list[str], gap_descriptions: list[str]) -> str:
    return _list_prompt(
        topic, "List 3 to 4 strengths of this research proposal.",
        _proposal_context(objectives, gap_descriptions),
    )


def reviewer_weaknesses_prompt(topic: str, objectives: list[str], gap_descriptions: list[str]) -> str:
    return _list_prompt(
        topic, "List 3 to 4 weaknesses or risks of this research proposal.",
        _proposal_context(objectives, gap_descriptions),
    )


def reviewer_missing_prompt(topic: str, objectives: list[str], gap_descriptions: list[str]) -> str:
    return _list_prompt(
        topic, "List 2 components missing from this research proposal.",
        _proposal_context(objectives, gap_descriptions),
    )


# ---------------------------------------------------------------------------
# Parsing helpers (shared by training-data builder and inference service)
# ---------------------------------------------------------------------------

_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def parse_string_list(raw: str, max_items: int = 6) -> list[str]:
    """Best-effort parse of a model response that should be a JSON array of
    strings. flan-t5 occasionally emits curly/smart quotes or an
    unbalanced closing bracket, which breaks strict JSON parsing even
    though the content itself (comma-separated quoted phrases) is fine —
    so before giving up, this tries progressively looser extraction
    rather than collapsing a good response into one giant string."""
    raw = raw.strip().translate(_SMART_QUOTES)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()][:max_items]
    except json.JSONDecodeError:
        pass

    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()][:max_items]
        except json.JSONDecodeError:
            pass

    # The array as a whole may be malformed (truncated, missing a closing
    # bracket/quote) while still containing well-formed "..." segments —
    # extract those directly rather than treating the whole blob as one item.
    quoted = re.findall(r'"([^"]{3,})"', raw)
    if quoted:
        return [q.strip().rstrip(",") for q in quoted if q.strip()][:max_items]

    # Last resort: split on newlines/semicolons/bullets.
    parts = [p.strip(" -*\t\"'") for p in raw.replace(";", "\n").split("\n")]
    return [p for p in parts if p][:max_items]
