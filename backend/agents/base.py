"""
Base class implementing the structural output validation policy shared by
all agents: strip fences -> validate against Pydantic schema -> on failure,
retry once with the raw invalid output + validation error -> on second
failure, fall back to an empty/default output with `parse_failed=True`
rather than crashing the session.

This is deliberately separate from the ACS quality-retry policy (see
services/collaboration.py), which operates one layer up in the
orchestrator, after a structurally valid output has been obtained.
"""
from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import settings
from services.llm import llm_service, _strip_code_fences
from utils.prompt_helpers import build_structural_retry_prompt

T = TypeVar("T", bound=BaseModel)


class StructuralRetryResult:
    def __init__(self, output: BaseModel, parse_failed: bool, raw_text: str):
        self.output = output
        self.parse_failed = parse_failed
        self.raw_text = raw_text


class BaseAgent:
    """Common structural-retry machinery. Subclasses provide the prompt and
    schema; this class handles the parse -> validate -> retry -> fallback
    flow uniformly."""

    def generate_validated(self, prompt: str, schema: Type[T]) -> StructuralRetryResult:
        raw_text = ""
        last_error = ""
        for attempt in range(settings.max_structural_retries_per_agent + 1):
            try:
                raw_text = llm_service.generate(prompt, agent=getattr(self, "name", None))
                cleaned = _strip_code_fences(raw_text)
                parsed = json.loads(cleaned)
                validated = schema.model_validate(parsed)
                return StructuralRetryResult(validated, parse_failed=False, raw_text=raw_text)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                if attempt < settings.max_structural_retries_per_agent:
                    prompt = build_structural_retry_prompt(prompt, raw_text, last_error)
                continue

        # Both attempts failed — do not crash the session. Return a default
        # (empty) instance and mark parse_failed so downstream agents and
        # the UI can react without hard-failing the pipeline.
        return StructuralRetryResult(schema(), parse_failed=True, raw_text=raw_text)
