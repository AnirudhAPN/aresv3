"""Agent 1 — Planner Agent."""
from __future__ import annotations

from config import settings
from agents.base import BaseAgent, StructuralRetryResult
from models.schemas import PlannerOutput
from prompts.planner_prompt import build_planner_prompt


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, topic: str) -> StructuralRetryResult:
        if settings.llm_provider == "local":
            from services.local_model import local_model_service
            data = local_model_service.run_planner(topic)
            return StructuralRetryResult(PlannerOutput(**data), parse_failed=False, raw_text="")

        prompt = build_planner_prompt(topic)
        return self.generate_validated(prompt, PlannerOutput)

    def run_with_retry_prompt(self, retry_prompt: str) -> StructuralRetryResult:
        return self.generate_validated(retry_prompt, PlannerOutput)


planner_agent = PlannerAgent()
