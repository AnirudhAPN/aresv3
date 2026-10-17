"""Agent 2 — Literature Agent.

Retrieval happens BEFORE any LLM call (services/literature_search.py).
The LLM is only ever used to synthesize a summary of real, already-fetched
abstracts — it is never asked to produce paper metadata.
"""
from __future__ import annotations

import json

from config import settings
from agents.base import BaseAgent
from models.schemas import LiteratureOutput, Paper
from prompts.literature_prompt import build_literature_synthesis_prompt
from services import similarity, url_verify
from services.citations import annotate_citations
from services.literature_search import literature_search_service
from services.llm import llm_service, _strip_code_fences


class LiteratureAgent(BaseAgent):
    name = "literature"

    def run(self, topic: str, keywords: list[str]) -> tuple[LiteratureOutput, bool]:
        papers, low_coverage = literature_search_service.search(keywords)

        if not papers:
            return LiteratureOutput(papers=[], low_coverage=True, keywords_used=keywords), False

        parse_failed = False
        if settings.llm_provider == "local":
            from services.local_model import local_model_service
            data = local_model_service.run_literature(topic, [p.model_dump() for p in papers])
            for paper, summary in zip(papers, data.get("summaries", [])):
                paper.summary = summary or paper.abstract[:280]
        else:
            # LLM synthesis step — constrained to the real retrieved abstracts.
            prompt = build_literature_synthesis_prompt(topic, [p.model_dump() for p in papers])
            try:
                raw = llm_service.generate(prompt, agent="literature")
                cleaned = _strip_code_fences(raw)
                parsed = json.loads(cleaned)
                summaries = parsed.get("summaries", [])
                for paper, summary in zip(papers, summaries):
                    paper.summary = summary
            except (json.JSONDecodeError, KeyError, TypeError):
                # Synthesis failed structurally — fall back to using the raw
                # abstract as the "summary" rather than failing the whole agent,
                # since the underlying retrieval (the part that matters for
                # verifiability) already succeeded.
                parse_failed = True
                for paper in papers:
                    if not paper.summary:
                        paper.summary = paper.abstract[:280]

        annotate_citations(papers)

        verified = url_verify.verify_urls_concurrently([p.url for p in papers])
        for paper in papers:
            paper.url_verified = verified.get(paper.url)

        similar_work = similarity.find_similar_existing_work(topic, papers)

        output = LiteratureOutput(
            papers=papers,
            low_coverage=low_coverage,
            keywords_used=keywords,
            similar_work=similar_work,
        )
        return output, parse_failed


literature_agent = LiteratureAgent()
