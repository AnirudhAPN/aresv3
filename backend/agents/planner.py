"""Agent 1 — Planner Agent."""
from __future__ import annotations

from agents.base import BaseAgent, StructuralRetryResult
from models.schemas import PlannerOutput
from prompts.planner_prompt import build_planner_prompt


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, topic: str) -> StructuralRetryResult:
        prompt = build_planner_prompt(topic)
        return self.generate_validated(prompt, PlannerOutput)

    def run_with_retry_prompt(self, retry_prompt: str) -> StructuralRetryResult:
        return self.generate_validated(retry_prompt, PlannerOutput)


planner_agent = PlannerAgent()
