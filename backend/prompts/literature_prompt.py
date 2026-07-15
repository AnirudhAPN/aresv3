"""Prompt template for the Literature Agent's synthesis step.

IMPORTANT: this prompt is only used AFTER real papers have been retrieved
via services/literature_search.py. It must never be used to generate paper
metadata itself — only to synthesize summaries of already-retrieved
abstracts.
"""
import json


def build_literature_synthesis_prompt(topic: str, papers: list[dict]) -> str:
    papers_block = json.dumps(
        [
            {
                "title": p["title"],
                "authors": p["authors"],
                "year": p["year"],
                "abstract": p["abstract"],
            }
            for p in papers
        ],
        indent=2,
    )
    return f"""You are the Literature Agent in a multi-agent research assistant.

Research topic: "{topic}"

Below are REAL papers retrieved from Semantic Scholar / arXiv, each with its
real abstract. You must synthesize a one-to-two sentence summary for EACH
paper explaining its relevance to the topic, using ONLY information present
in its abstract below. Do NOT invent findings not stated in the abstract.
Do NOT invent any paper not in this list.

Papers:
{papers_block}

Respond with ONLY valid JSON of this exact shape (same order as input, one
summary per paper, no prose outside the JSON):
{{
  "summaries": ["summary for paper 1", "summary for paper 2", "..."]
}}
"""
