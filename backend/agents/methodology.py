"""Agent 4 — Methodology Agent (whitelist-grounded)."""
from __future__ import annotations

import json
from pathlib import Path

from agents.base import BaseAgent, StructuralRetryResult
from models.schemas import MethodologyOutput
from prompts.methodology_prompt import build_methodology_prompt
from services import url_verify
from services.kaggle_links import resolve_kaggle_url
from utils.text import strip_leading_ordinal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_whitelist(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    with open(path, "r") as f:
        return json.load(f)


class MethodologyAgent(BaseAgent):
    name = "methodology"

    def __init__(self) -> None:
        self.known_datasets = _load_whitelist("known_datasets.json")
        self.known_algorithms = _load_whitelist("known_algorithms.json")

    def run(self, topic: str, research_questions: list[str]) -> StructuralRetryResult:
        prompt = build_methodology_prompt(
            topic, research_questions, self.known_datasets, self.known_algorithms
        )
        result = self.generate_validated(prompt, MethodologyOutput)

        # Defense in depth: re-verify against the whitelist in code, in case
        # the model marked verified=true for something not actually on it.
        if isinstance(result.output, MethodologyOutput):
            dataset_names = {d["name"].lower() for d in self.known_datasets}
            algorithm_names = {a["name"].lower() for a in self.known_algorithms}
            for rec in result.output.datasets:
                rec.verified = rec.name.lower() in dataset_names
                rec.kaggle_url = resolve_kaggle_url(rec.name, self.known_datasets)
            for rec in result.output.algorithms:
                rec.verified = rec.name.lower() in algorithm_names

            items = result.output.datasets + result.output.algorithms
            urls = [rec.url for rec in items if rec.url]
            checked = url_verify.verify_urls_concurrently(urls)
            for rec in items:
                rec.url_verified = checked.get(rec.url) if rec.url else None

            # Defense in depth: the UI and exports each apply their own
            # sequential numbering to implementation_steps — strip any
            # ordinal the model embedded itself so steps never render
            # double-numbered or out of sequence.
            result.output.implementation_steps = [
                strip_leading_ordinal(s) for s in result.output.implementation_steps
            ]

        return result


methodology_agent = MethodologyAgent()
