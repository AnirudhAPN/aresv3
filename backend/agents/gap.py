"""Agent 3 — Gap Analysis Agent (evidence-bounded)."""
from __future__ import annotations

from difflib import SequenceMatcher

from config import settings
from agents.base import BaseAgent, StructuralRetryResult
from models.schemas import GapCitation, GapOutput, LiteratureOutput
from prompts.gap_prompt import build_gap_prompt

CITATION_MATCH_THRESHOLD = 0.6


def _resolve_citations(titles: list[str], papers: list[dict]) -> list[GapCitation]:
    """Maps each `based_on_papers` title string back to the real retrieved
    paper it names, so the UI can link to an actual source instead of an
    unclickable string. Neither provider is asked to emit a URL directly —
    a generated URL is exactly the kind of thing that gets fabricated —
    resolution is always done here, against real retrieval results, the
    same way novelty_score is centrally overridden in reviewer.py rather
    than trusted from either provider."""
    resolved = []
    for title in titles:
        norm = title.strip().lower()
        best_paper, best_score = None, 0.0
        for p in papers:
            p_title = (p.get("title") or "").strip().lower()
            if not p_title:
                continue
            score = 1.0 if norm == p_title else SequenceMatcher(None, norm, p_title).ratio()
            if score > best_score:
                best_score, best_paper = score, p
        if best_paper and best_score >= CITATION_MATCH_THRESHOLD:
            resolved.append(
                GapCitation(
                    title=best_paper.get("title", title),
                    url=best_paper.get("url"),
                    source=best_paper.get("source"),
                    year=best_paper.get("year"),
                    matched=True,
                )
            )
        else:
            resolved.append(GapCitation(title=title, matched=False))
    return resolved


class GapAgent(BaseAgent):
    name = "gap"

    def run(self, topic: str, literature_output: LiteratureOutput) -> StructuralRetryResult:
        papers = [p.model_dump() for p in literature_output.papers]

        if settings.llm_provider == "local":
            from services.local_model import local_model_service
            data = local_model_service.run_gap(topic, papers, literature_output.low_coverage)
            result = StructuralRetryResult(GapOutput(**data), parse_failed=False, raw_text="")
        else:
            prompt = build_gap_prompt(topic, papers, literature_output.low_coverage)
            result = self.generate_validated(prompt, GapOutput)

        # Enforce propagation of low_coverage -> low_confidence_gaps even if
        # the model didn't set it, since this is a hard rule, not a
        # suggestion.
        if isinstance(result.output, GapOutput):
            if literature_output.low_coverage:
                result.output.low_confidence_gaps = True
            for gap in result.output.gaps:
                gap.citations = _resolve_citations(gap.based_on_papers, papers)

        return result


gap_agent = GapAgent()
