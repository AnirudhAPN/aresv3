"""Agent 3 — Gap Analysis Agent (evidence-bounded)."""
from __future__ import annotations

from agents.base import BaseAgent, StructuralRetryResult
from models.schemas import GapOutput, LiteratureOutput
from prompts.gap_prompt import build_gap_prompt


class GapAgent(BaseAgent):
    name = "gap"

    def run(self, topic: str, literature_output: LiteratureOutput) -> StructuralRetryResult:
        papers = [p.model_dump() for p in literature_output.papers]
        prompt = build_gap_prompt(topic, papers, literature_output.low_coverage)
        result = self.generate_validated(prompt, GapOutput)

        # Enforce propagation of low_coverage -> low_confidence_gaps even if
        # the model didn't set it, since this is a hard rule, not a
        # suggestion.
        if literature_output.low_coverage and isinstance(result.output, GapOutput):
            result.output.low_confidence_gaps = True

        return result


gap_agent = GapAgent()
