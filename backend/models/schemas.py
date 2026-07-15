"""
Pydantic models shared across the whole pipeline: per-agent outputs, the
LangGraph shared state, and DB-facing session models.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent 1 — Planner
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    research_objectives: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    research_scope: str = ""


# ---------------------------------------------------------------------------
# Agent 2 — Literature (retrieval-first, never hallucinated)
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    url: str
    citation_count: Optional[int] = None
    source: Literal["semantic_scholar", "arxiv"]
    abstract: str = ""
    summary: str = ""          # Gemini's synthesis of the REAL abstract
    verified: bool = True      # always True — retrieved, not generated
    url_verified: Optional[bool] = None   # None = not yet checked, True/False = HTTP-checked
    citation_apa: str = ""
    citation_bibtex: str = ""


class SimilarWorkMatch(BaseModel):
    """A retrieved paper whose content is close enough to the proposed
    research topic that it may already cover the same work."""
    paper_title: str
    paper_url: str
    similarity: float


class LiteratureOutput(BaseModel):
    papers: list[Paper] = Field(default_factory=list)
    low_coverage: bool = False
    keywords_used: list[str] = Field(default_factory=list)
    similar_work: Optional[SimilarWorkMatch] = None


# ---------------------------------------------------------------------------
# Agent 3 — Gap Analysis (evidence-bounded)
# ---------------------------------------------------------------------------

class ResearchGap(BaseModel):
    gap_description: str
    based_on_papers: list[str] = Field(default_factory=list)
    confidence_note: str = ""


class GapOutput(BaseModel):
    current_trends: list[str] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)
    future_directions: list[str] = Field(default_factory=list)
    low_confidence_gaps: bool = False


# ---------------------------------------------------------------------------
# Agent 4 — Methodology (whitelist-grounded)
# ---------------------------------------------------------------------------

class DatasetRec(BaseModel):
    name: str
    url: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    url_verified: Optional[bool] = None   # None = not yet checked / no URL
    kaggle_url: Optional[str] = None      # the corresponding or most-relevant Kaggle page


class MethodologyOutput(BaseModel):
    datasets: list[DatasetRec] = Field(default_factory=list)
    algorithms: list[DatasetRec] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    unverified_suggestions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 5 — Reviewer (rule-checks + qualitative critique)
# ---------------------------------------------------------------------------

class ReviewerOutput(BaseModel):
    rule_check_results: dict[str, bool] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_components: list[str] = Field(default_factory=list)
    novelty_score: float = 0.0
    quality_score: float = 0.0
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared LangGraph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    session_id: str
    topic: str
    planner_output: Optional[PlannerOutput]
    literature_output: Optional[LiteratureOutput]
    gap_output: Optional[GapOutput]
    methodology_output: Optional[MethodologyOutput]
    reviewer_output: Optional[ReviewerOutput]
    acs_scores: dict[str, float]
    acs_components: dict[str, dict[str, float]]
    retry_counts: dict[str, int]
    structural_retry_counts: dict[str, int]
    parse_failures: dict[str, bool]
    flags: list[str]
    current_agent: str
    completed_agents: list[str]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    agent_updated: Optional[str] = None
    created_at: float


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# AI Assistant — research planning / suggestions / workflow guidance.
# Distinct from ChatMessage/ChatRequest above, which refine one completed
# research proposal; the assistant is a single, session-independent
# conversation that helps *before* and *around* running the pipeline.
# ---------------------------------------------------------------------------

class AssistantMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    suggested_topic: Optional[str] = None
    created_at: float


class AssistantChatRequest(BaseModel):
    message: str


class AssistantChatResponse(BaseModel):
    reply: str
    suggested_topic: Optional[str] = None


# ---------------------------------------------------------------------------
# DB-facing models (API responses)
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str


class SimilarSession(BaseModel):
    session_id: str
    topic: str
    similarity: float


class ResearchStartResponse(BaseModel):
    session_id: str
    status: str
    similar_sessions: list[SimilarSession] = Field(default_factory=list)


class ResearchStatusResponse(BaseModel):
    session_id: str
    status: Literal["running", "completed", "failed"]
    current_agent: Optional[str] = None
    completed_agents: list[str] = Field(default_factory=list)
    acs_scores: dict[str, float] = Field(default_factory=dict)
    acs_components: dict[str, dict[str, float]] = Field(default_factory=dict)


class ResearchSession(BaseModel):
    session_id: str
    user_id: str = "default"
    topic: str
    status: str
    created_at: datetime
    planner_output: Optional[PlannerOutput] = None
    literature_output: Optional[LiteratureOutput] = None
    gap_output: Optional[GapOutput] = None
    methodology_output: Optional[MethodologyOutput] = None
    reviewer_output: Optional[ReviewerOutput] = None
    acs_scores: dict[str, float] = Field(default_factory=dict)
    acs_components: dict[str, dict[str, float]] = Field(default_factory=dict)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    parse_failures: dict[str, bool] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    agent_updated: Optional[str] = None
    session: dict
