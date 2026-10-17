"""
Central configuration for ARES backend.

All tunables that affect the research pipeline's behavior live here so they
can be adjusted (or ablated during evaluation) without touching business
logic elsewhere in the codebase.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ACSWeights:
    """Weights for the Adaptive Collaboration Score formula.

    ACS = confidence_weight * confidence
        + evidence_weight * evidence_quality
        + completion_weight * task_completion

    Weights should sum to 1.0. Kept configurable (not hardcoded in the
    formula) so they can be tuned/ablated as part of the evaluation.
    """
    confidence_weight: float = float(os.getenv("ACS_CONFIDENCE_WEIGHT", 0.40))
    evidence_weight: float = float(os.getenv("ACS_EVIDENCE_WEIGHT", 0.35))
    completion_weight: float = float(os.getenv("ACS_COMPLETION_WEIGHT", 0.25))

    def __post_init__(self):
        total = self.confidence_weight + self.evidence_weight + self.completion_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ACS weights must sum to 1.0, got {total}")


@dataclass
class Settings:
    # --- LLM providers ---
    # "api": use Gemini (generate) + Anthropic (judge), the original design.
    # "local": use the locally fine-tuned per-agent models in
    #   models/finetuned/<agent>/adapter — no network calls, no API keys,
    #   no quota. See scripts/finetune_agent.py and services/local_model.py.
    llm_provider: str = os.getenv("LLM_PROVIDER", "api")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Judge model: ideally a DIFFERENT provider/model from the generator to
    # reduce same-family bias in confidence scoring and reviewer critique.
    judge_provider: str = os.getenv("JUDGE_PROVIDER", "anthropic")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # --- External literature APIs ---
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    arxiv_base_url: str = "https://export.arxiv.org/api/query"

    # --- Retrieval policy ---
    min_papers_for_full_coverage: int = 3
    max_papers_returned: int = 8
    literature_cache_ttl_days: int = 30

    # --- ACS / retry policy ---
    acs_weights: ACSWeights = field(default_factory=ACSWeights)
    acs_retry_threshold: float = float(os.getenv("ACS_RETRY_THRESHOLD", 0.6))
    max_acs_retries_per_agent: int = 1
    max_structural_retries_per_agent: int = 1

    # --- Database ---
    database_path: str = os.getenv("DATABASE_PATH", "ares.db")

    # --- Server ---
    cors_origins: tuple = ("http://localhost:5173", "http://localhost:3000")


settings = Settings()
