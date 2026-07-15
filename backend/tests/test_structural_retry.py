"""
Tests the structural-retry policy (agents/base.py): a deliberately
malformed LLM response should trigger exactly one retry with the raw
invalid output + validation error, and if that also fails, the agent must
fall back to `parse_failed=True` with a default/empty output rather than
crashing the session.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base import BaseAgent
from models.schemas import PlannerOutput
from services.llm import llm_service


def test_structural_retry_recovers_on_second_attempt():
    responses = [
        "This is not JSON at all, sorry!",  # attempt 1: fails to parse
        '{"research_objectives": ["Obj 1"], "research_questions": ["Q1"], '
        '"keywords": ["k1"], "research_scope": "Scope"}',  # attempt 2: valid
    ]

    with patch.object(llm_service, "generate", side_effect=responses) as mock_gen:
        agent = BaseAgent()
        result = agent.generate_validated("dummy prompt", PlannerOutput)

    assert result.parse_failed is False
    assert isinstance(result.output, PlannerOutput)
    assert result.output.research_objectives == ["Obj 1"]
    assert mock_gen.call_count == 2  # exactly one retry was used


def test_structural_retry_falls_back_after_exhausting_retries():
    # Both attempts return invalid JSON -> must not raise, must mark
    # parse_failed=True and return a default/empty schema instance.
    responses = ["not json", "still not json"]

    with patch.object(llm_service, "generate", side_effect=responses) as mock_gen:
        agent = BaseAgent()
        result = agent.generate_validated("dummy prompt", PlannerOutput)

    assert result.parse_failed is True
    assert isinstance(result.output, PlannerOutput)
    assert result.output.research_objectives == []  # default/empty fallback
    assert mock_gen.call_count == 2  # initial attempt + 1 retry, then stop


def test_structural_retry_handles_schema_validation_error():
    # Valid JSON but wrong shape (extra required-looking mismatch handled
    # gracefully by Pydantic defaults) followed by a fully valid response.
    responses = [
        '{"research_objectives": "not a list"}',  # wrong type -> ValidationError
        '{"research_objectives": ["Obj 1"], "research_questions": [], '
        '"keywords": [], "research_scope": ""}',
    ]
    with patch.object(llm_service, "generate", side_effect=responses):
        agent = BaseAgent()
        result = agent.generate_validated("dummy prompt", PlannerOutput)

    assert result.parse_failed is False
    assert result.output.research_objectives == ["Obj 1"]


if __name__ == "__main__":
    test_structural_retry_recovers_on_second_attempt()
    test_structural_retry_falls_back_after_exhausting_retries()
    test_structural_retry_handles_schema_validation_error()
    print("All structural retry tests passed.")
