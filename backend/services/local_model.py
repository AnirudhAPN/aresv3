"""Inference for the locally fine-tuned per-agent models.

Each of the 5 ARES agents (planner, literature, gap, methodology, reviewer)
has its own small LoRA adapter, trained on top of a shared frozen base model
(google/flan-t5-small) by scripts/finetune_agent.py, saved under
models/finetuned/<agent>/adapter.

IMPORTANT — task decomposition: the small local model is NOT asked to
produce one giant nested JSON object in a single call the way the Gemini
API path is. That collapsed to invalid JSON even as training loss dropped,
because "get 4-6 fields AND nested objects AND JSON syntax right in one
shot" is a fair ask for a 100B+ parameter model and an unfair one for a
77M-parameter model fine-tuned on a few dozen examples. Instead, each
agent's output is assembled from several small single-purpose generations
(services/local_decomposed.py builds the prompts) plus fields that are
constructed deterministically in code rather than generated at all:
  - dataset/algorithm whitelist selection -> embedding similarity, not
    generation (guarantees whitelist-grounded, correctly-typed results)
  - gap evidence citations / confidence notes -> built from the actual
    retrieved paper list, not invented
  - reviewer novelty/quality scores -> derived from rule-check pass rate,
    not asked of a model that reliably outputs calibrated floats

No network access, no API key, no quota anywhere in this file.
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from services import local_decomposed as ld

BASE_MODEL = "google/flan-t5-small"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "finetuned"
MAX_INPUT_TOKENS = 768
MAX_TARGET_TOKENS = 512

AGENTS = ["planner", "literature", "gap", "methodology", "reviewer"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LocalModelService:
    """Lazy-loading, per-agent LoRA adapter inference."""

    def __init__(self) -> None:
        self._models: dict[str, PeftModel] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}

    def is_trained(self, agent: str) -> bool:
        return (MODELS_DIR / agent / "adapter").exists()

    def _load(self, agent: str) -> tuple[PeftModel, AutoTokenizer]:
        if agent in self._models:
            return self._models[agent], self._tokenizers[agent]

        adapter_path = MODELS_DIR / agent / "adapter"
        if not adapter_path.exists():
            raise RuntimeError(
                f"No local fine-tuned model found for agent '{agent}' at {adapter_path}. "
                f"Run `python scripts/finetune_agent.py --agent {agent}` first."
            )

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model = model.to(DEVICE)
        model.eval()

        self._models[agent] = model
        self._tokenizers[agent] = tokenizer
        return model, tokenizer

    def generate(self, agent: str, prompt: str, max_new_tokens: int = 96, min_new_tokens: int = 0) -> str:
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent '{agent}' — expected one of {AGENTS}")
        model, tokenizer = self._load(agent)
        inputs = tokenizer(prompt, max_length=MAX_INPUT_TOKENS, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            # A small fine-tuned model on limited data tends to loop (repeat
            # the same list item several times) under greedy decoding;
            # repetition_penalty suppresses that. Tried lowering it and
            # forcing min_new_tokens to fix under-length lists — empirically
            # WORSE: forcing the model past where it naturally wants to stop
            # pushes it into degenerate babbling rather than more real
            # content, since it has no strong signal for what comes next.
            # Kept at the setting that empirically produced the most valid,
            # least-garbled JSON.
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min(min_new_tokens, max_new_tokens) if min_new_tokens else 0,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    def _pick_whitelist(self, topic: str, whitelist: list[dict], top_k: int) -> list[dict]:
        """Deterministic dataset/algorithm selection via embedding
        similarity — never asks the small model to invent a name/URL."""
        if not whitelist:
            return []
        from services.embeddings import get_embedder

        embedder = get_embedder()
        topic_emb = embedder.encode([topic])
        texts = [f"{w['name']}. {w.get('description', '')}" for w in whitelist]
        whitelist_emb = embedder.encode(texts)
        sims = cosine_similarity(topic_emb, whitelist_emb)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {
                "name": whitelist[i]["name"],
                "url": whitelist[i].get("url"),
                "description": whitelist[i].get("description"),
                "verified": True,
            }
            for i in top_idx
        ]

    # -- Per-agent orchestration -------------------------------------------

    def run_planner(self, topic: str) -> dict:
        objectives = ld.parse_string_list(self.generate("planner", ld.planner_objectives_prompt(topic)))
        questions = ld.parse_string_list(self.generate("planner", ld.planner_questions_prompt(topic)))
        raw_keywords = ld.parse_string_list(self.generate("planner", ld.planner_keywords_prompt(topic)), max_items=5)
        keywords = self._filter_relevant_keywords(topic, raw_keywords)
        scope = self.generate("planner", ld.planner_scope_prompt(topic), max_new_tokens=160).strip()
        return {
            "research_objectives": objectives,
            "research_questions": questions,
            "keywords": keywords,
            "research_scope": scope,
        }

    def _filter_relevant_keywords(
        self, topic: str, keywords: list[str], top_k: int = 3, min_similarity: float = 0.20
    ) -> list[str]:
        """Keeps only the keywords most semantically related to the topic
        before they reach the REAL literature search. Small fine-tuned
        models occasionally hallucinate a plausible-sounding but off-topic
        keyword (e.g. "robot-aware forensic training" for a cyberbullying
        topic) — since retrieval is genuine, not mocked, one bad keyword
        can drag in unrelated real papers and cascade through gap/
        methodology grounding.

        An absolute similarity floor alone isn't enough: an off-topic
        keyword can still score high enough to pass a fixed threshold while
        being clearly worse than the topic's other keywords (observed case:
        "speech-based forensic detection" scored 0.25 for a cyberbullying
        topic — above a naive 0.20 cutoff, but nowhere near "cyberbullying
        detection" at 0.76 — and still hijacked retrieval with unrelated
        speech-pathology papers). Ranking by similarity and keeping only
        the top few fixes that: a mediocre keyword only survives if the
        model didn't generate anything better.

        Falls back to the topic itself if filtering would leave nothing.
        """
        if not keywords:
            return [topic]
        from services.embeddings import get_embedder

        embedder = get_embedder()
        topic_emb = embedder.encode([topic])
        kw_emb = embedder.encode(keywords)
        sims = cosine_similarity(topic_emb, kw_emb)[0]
        ranked = sorted(zip(keywords, sims), key=lambda pair: pair[1], reverse=True)
        relevant = [kw for kw, sim in ranked[:top_k] if sim >= min_similarity]
        if not relevant:
            return [topic]

        # literature_search.py joins ALL keywords into one combined query
        # string (not searched individually) — so even a single surviving
        # mediocre keyword can still dominate ranking on real retrieval
        # APIs (observed: "speech-based forensic detection" pulled in
        # unrelated speech-pathology papers for a cyberbullying topic).
        # Anchoring the query with the topic itself, not just whatever the
        # small model generated, keeps retrieval grounded even when
        # keyword generation quality is mediocre.
        return [topic] + relevant

    def run_literature(self, topic: str, papers: list[dict]) -> dict:
        summaries = []
        for p in papers:
            prompt = ld.literature_summary_prompt(topic, p.get("title", ""), p.get("abstract", ""))
            summaries.append(self.generate("literature", prompt, max_new_tokens=64).strip())
        return {"summaries": summaries}

    def run_gap(self, topic: str, papers: list[dict], low_coverage: bool) -> dict:
        trends = ld.parse_string_list(self.generate("gap", ld.gap_trends_prompt(topic, papers)))
        bare_descriptions = ld.parse_string_list(
            self.generate("gap", ld.gap_descriptions_prompt(topic, papers)), max_items=3
        )
        future = ld.parse_string_list(self.generate("gap", ld.gap_future_directions_prompt(topic)))

        titles = [p.get("title", "") for p in papers if p.get("title")]
        n = len(papers)
        gaps = [
            {
                "gap_description": ld.GAP_DESCRIPTION_PREFIX + desc,
                "based_on_papers": titles[:2],
                "confidence_note": f"based on {n} retrieved papers" if n else "based on limited retrieved papers",
            }
            for desc in bare_descriptions
        ]

        return {
            "current_trends": trends,
            "gaps": gaps,
            "future_directions": future,
            "low_confidence_gaps": low_coverage,
        }

    def run_methodology(
        self, topic: str, research_questions: list[str], known_datasets: list[dict], known_algorithms: list[dict]
    ) -> dict:
        metrics = ld.parse_string_list(self.generate("methodology", ld.methodology_metrics_prompt(topic)))
        steps = ld.parse_string_list(
            self.generate("methodology", ld.methodology_steps_prompt(topic), max_new_tokens=160), max_items=6
        )
        unverified = ld.parse_string_list(
            self.generate("methodology", ld.methodology_unverified_prompt(topic)), max_items=3
        )
        query = f"{topic}. {' '.join(research_questions[:2])}" if research_questions else topic
        datasets = self._pick_whitelist(query, known_datasets, top_k=1)
        algorithms = self._pick_whitelist(query, known_algorithms, top_k=2)

        return {
            "datasets": datasets,
            "algorithms": algorithms,
            "evaluation_metrics": metrics,
            "implementation_steps": steps,
            "unverified_suggestions": unverified,
        }

    def run_reviewer(
        self,
        topic: str,
        objectives: list[str],
        gap_descriptions: list[str],
        rule_check_results: dict[str, bool],
    ) -> dict:
        strengths = ld.parse_string_list(
            self.generate("reviewer", ld.reviewer_strengths_prompt(topic, objectives, gap_descriptions))
        )
        weaknesses = ld.parse_string_list(
            self.generate("reviewer", ld.reviewer_weaknesses_prompt(topic, objectives, gap_descriptions))
        )
        missing = ld.parse_string_list(
            self.generate("reviewer", ld.reviewer_missing_prompt(topic, objectives, gap_descriptions)),
            max_items=2,
        )

        total = len(rule_check_results) or 1
        passed = sum(1 for v in rule_check_results.values() if v)
        quality_score = round(0.5 + 0.4 * (passed / total), 2)

        # novelty_score is intentionally NOT computed here — agents/reviewer.py
        # overrides it uniformly (for both providers) with a real,
        # evidence-grounded score via services.similarity.compute_novelty_score,
        # rather than trusting a small model's ungrounded self-assessment.
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "missing_components": missing,
            "quality_score": quality_score,
        }


local_model_service = LocalModelService()
