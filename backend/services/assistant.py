"""
AI Assistant Agent — research planning, suggestions, and workflow guidance.

Deliberately distinct from services/chat.py: that module refines ONE
already-completed research proposal (re-running a specific pipeline agent).
This assistant is a single, session-independent conversation that helps
*before* and *around* running the pipeline — narrowing a vague topic into a
researchable one, suggesting angles/keywords, and explaining how ARES's
five-agent pipeline, ACS scoring, and chat-refinement workflow work.

Same evidence-grounding rule as the rest of the app applies here: this
agent must never claim to know specific real papers, datasets, or citations
for the user's topic from its own training knowledge — that's exactly the
hallucination risk the Literature/Methodology agents are built to avoid by
retrieving real, verifiable sources instead. The assistant's job is
methodological (framing, scope, keywords, what to expect from the
pipeline), not literature synthesis.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from services.llm import llm_service, _strip_code_fences
from services.memory import memory_service

MAX_HISTORY_MESSAGES = 12

ASSISTANT_SYSTEM_PROMPT = """You are the ARES Research Assistant — a planning and workflow guide inside
an AI-powered multi-agent research proposal tool. You help with exactly three things:

1. RESEARCH PLANNING: help the user turn a vague idea into a well-scoped, researchable topic —
   ask a clarifying question if the topic is too broad or ambiguous, suggest how to narrow scope,
   and propose a refined topic phrasing when you have enough to do so.
2. SUGGESTIONS: propose related angles, sub-questions, or search keywords the user could explore.
3. WORKFLOW GUIDANCE: explain how ARES works when asked — a topic is run through five agents in
   sequence (Planner, Literature, Gap Analysis, Methodology, Reviewer), each output is scored by an
   Adaptive Collaboration Score (confidence + evidence quality + task completion, 0-1); anything
   scoring below 0.60 is automatically retried once, then flagged "low confidence" rather than
   looping forever. Once a proposal completes, the user can keep refining any section by chatting
   about it on the results page, and export the result as Markdown, plain text, or PDF.

HARD RULE: you do not have access to real literature search. NEVER invent, name, or describe
specific papers, authors, datasets, or citations as if they were real — that is exactly the
hallucination risk this tool exists to prevent, and it is the Literature/Methodology agents' job
(they retrieve from Semantic Scholar/arXiv and a verified whitelist, not from your own training
knowledge). If the user wants real sources, tell them to run the topic through the pipeline instead.

Keep replies to 2-4 sentences, conversational, and concrete — no filler.

Respond with ONLY a JSON object of this exact shape, no other text:
{"reply": "<your conversational reply>", "suggested_topic": "<a concrete, well-scoped research topic if you have one to propose right now, else null>"}
"""


def _build_prompt(history: list[dict[str, Any]], message: str) -> str:
    recent = history[-MAX_HISTORY_MESSAGES:]
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    transcript_block = f"\nConversation so far:\n{transcript}\n" if transcript else ""
    return f"{ASSISTANT_SYSTEM_PROMPT}\n{transcript_block}\nuser: {message}\n"


def chat(message: str, user_id: str = "default") -> dict[str, Optional[str]]:
    """Returns {"reply": str, "suggested_topic": str | None} and persists
    both the user's message and the assistant's reply."""
    history = memory_service.get_assistant_history(user_id=user_id)
    memory_service.add_assistant_message("user", message, user_id=user_id)

    prompt = _build_prompt(history, message)
    raw = ""
    suggested_topic: Optional[str] = None
    try:
        raw = llm_service.generate(prompt, temperature=0.5)
        parsed = json.loads(_strip_code_fences(raw))
        reply = str(parsed.get("reply", "")).strip()
        suggested_topic = parsed.get("suggested_topic") or None
        if not reply:
            raise ValueError("empty reply")
    except Exception:
        # Structural parsing failed — fall back to the raw model text
        # rather than dropping the turn, consistent with how the rest of
        # the app degrades gracefully instead of crashing on bad JSON.
        reply = raw.strip() or "I couldn't quite process that — could you rephrase your research idea or question?"
        suggested_topic = None

    memory_service.add_assistant_message("assistant", reply, suggested_topic=suggested_topic, user_id=user_id)
    return {"reply": reply, "suggested_topic": suggested_topic}
