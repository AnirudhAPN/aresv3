"""
Thin wrappers around the LLM providers used by ARES.

Two distinct roles are kept separate on purpose:
  - `generate()` calls the GENERATING model (Gemini) that produces each
    agent's content.
  - `judge()` calls a SEPARATE model (default: Anthropic/Claude) used for
    confidence scoring and Part B of the Reviewer Agent's critique. Using a
    different model family for judging reduces the circularity of a model
    grading its own output.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types as genai_types
import anthropic

from config import settings


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences the model may add."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# A provider suggesting a wait longer than this means "come back much
# later" (e.g. a daily quota, not a transient rate limit) — better to fail
# the agent call fast than block a background pipeline task for minutes.
_MAX_HONORED_RETRY_DELAY_SECONDS = 90.0


def _gemini_retry_delay_seconds(exc: Exception) -> float | None:
    """Read Gemini's own suggested wait time out of the structured error
    body (google.rpc.RetryInfo), instead of guessing with blind exponential
    backoff — the API tells us exactly how long to wait."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    for entry in details.get("error", {}).get("details", []):
        if str(entry.get("@type", "")).endswith("RetryInfo"):
            match = re.match(r"([\d.]+)", str(entry.get("retryDelay", "")))
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
    return None


def _anthropic_retry_delay_seconds(exc: Exception) -> float | None:
    """Read the standard HTTP Retry-After header Anthropic sends on 429s."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    if header is None:
        return None
    try:
        return float(header)
    except (TypeError, ValueError):
        return None


class LLMService:
    """Facade for all LLM calls used across agents."""

    def __init__(self) -> None:
        self._gemini_client: genai.Client | None = None
        self._anthropic_client: anthropic.Anthropic | None = None

    def _ensure_gemini(self) -> genai.Client:
        if self._gemini_client is None:
            if not settings.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to backend/.env before "
                    "running the pipeline."
                )
            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
        return self._gemini_client

    def _ensure_anthropic(self) -> anthropic.Anthropic:
        if self._anthropic_client is None:
            if not settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to backend/.env, or "
                    "change JUDGE_PROVIDER to reuse Gemini (not recommended: "
                    "same-family bias)."
                )
            self._anthropic_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key
            )
        return self._anthropic_client

    # -- Generation (Gemini) -------------------------------------------------

    def generate(self, prompt: str, temperature: float = 0.4, max_retries: int = 3) -> str:
        """Call the generating model and return raw text.
        Retries automatically on 429 rate-limit errors with exponential backoff."""
        client = self._ensure_gemini()
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=temperature),
                )
                return response.text
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    server_delay = _gemini_retry_delay_seconds(exc)
                    if server_delay is not None and server_delay > _MAX_HONORED_RETRY_DELAY_SECONDS:
                        logging.warning(
                            "Gemini rate-limited with a long suggested wait (%.0fs, likely a daily quota) "
                            "— giving up rather than blocking.", server_delay,
                        )
                        raise
                    if attempt < max_retries:
                        wait = server_delay if server_delay is not None else min(2 ** attempt * 15, 120)
                        logging.warning(
                            "Gemini rate-limited (attempt %d/%d). Retrying in %.0fs...",
                            attempt + 1, max_retries + 1, wait,
                        )
                        time.sleep(wait)
                        continue
                    logging.warning("Gemini rate-limited (attempt %d/%d). No retries left.", attempt + 1, max_retries + 1)
                raise

    def generate_json(self, prompt: str, temperature: float = 0.3) -> dict[str, Any]:
        """Call the generating model and parse the response as JSON.

        Raises `json.JSONDecodeError` on failure so callers can drive the
        structural-retry policy explicitly rather than silently swallowing
        errors here.
        """
        raw = self.generate(prompt, temperature=temperature)
        cleaned = _strip_code_fences(raw)
        return json.loads(cleaned)

    # -- Judging (separate model, default Anthropic) -------------------------

    def judge(self, prompt: str, temperature: float = 0.0, max_retries: int = 3) -> str:
        """Call the judge model (independent of the generator) for
        confidence scoring or qualitative critique.
        Retries automatically on rate-limit errors with exponential backoff."""
        if settings.judge_provider == "anthropic":
            client = self._ensure_anthropic()
            for attempt in range(max_retries + 1):
                try:
                    resp = client.messages.create(
                        model=settings.judge_model,
                        max_tokens=1024,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return "".join(
                        block.text for block in resp.content if block.type == "text"
                    )
                except Exception as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        server_delay = _anthropic_retry_delay_seconds(exc)
                        if server_delay is not None and server_delay > _MAX_HONORED_RETRY_DELAY_SECONDS:
                            logging.warning(
                                "Anthropic rate-limited with a long suggested wait (%.0fs) "
                                "— giving up rather than blocking.", server_delay,
                            )
                            raise
                        if attempt < max_retries:
                            wait = server_delay if server_delay is not None else min(2 ** attempt * 15, 120)
                            logging.warning(
                                "Anthropic rate-limited (attempt %d/%d). Retrying in %.0fs...",
                                attempt + 1, max_retries + 1, wait,
                            )
                            time.sleep(wait)
                            continue
                        logging.warning("Anthropic rate-limited (attempt %d/%d). No retries left.", attempt + 1, max_retries + 1)
                    raise
        # Fallback: reuse the generator (logged loudly — not recommended).
        return self.generate(prompt, temperature=temperature)

    def judge_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        raw = self.judge(prompt, temperature=temperature)
        cleaned = _strip_code_fences(raw)
        return json.loads(cleaned)


llm_service = LLMService()
