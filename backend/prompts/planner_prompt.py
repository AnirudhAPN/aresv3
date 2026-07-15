"""Prompt template for the Planner Agent."""

PLANNER_SCHEMA_HINT = """{
  "research_objectives": ["..."],
  "research_questions": ["..."],
  "keywords": ["..."],
  "research_scope": "..."
}"""


def build_planner_prompt(topic: str) -> str:
    return f"""You are the Planner Agent in a multi-agent research assistant.

Given a research topic, produce a structured research plan.

Topic: "{topic}"

Requirements:
- 3-5 concrete research objectives.
- 3-5 research questions the eventual proposal should answer.
- 6-10 keywords/phrases suitable for searching academic literature APIs
  (Semantic Scholar, arXiv). These must be specific search terms, not vague
  restatements of the topic.
- A short (2-4 sentence) research scope statement defining boundaries
  (what is in scope / out of scope).

Respond with ONLY valid JSON matching this exact shape, no prose, no markdown fences:
{PLANNER_SCHEMA_HINT}
"""
