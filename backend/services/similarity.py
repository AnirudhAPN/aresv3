"""
"Does similar research already exist?" checks.

Two independent signals, both evidence-bounded (never asks an LLM to judge
novelty from its own prior knowledge — that's exactly the hallucination
risk ARES is built to avoid):

  - find_similar_existing_work: compares the proposed topic against the
    REAL papers the Literature Agent already retrieved, via embedding
    similarity. A high match means a retrieved paper appears to already
    cover the same work.
  - Cross-session duplicate checks (has the user already researched this
    in ARES before) live in services/memory.py, since they only need
    simple string similarity over locally stored topics — no embedder.
"""
from __future__ import annotations

from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import Paper, SimilarWorkMatch
from services.embeddings import get_embedder

SIMILARITY_THRESHOLD = 0.80


def find_similar_existing_work(
    topic: str, papers: list[Paper], threshold: float = SIMILARITY_THRESHOLD
) -> Optional[SimilarWorkMatch]:
    if not topic.strip() or not papers:
        return None

    embedder = get_embedder()
    topic_embedding = embedder.encode([topic])
    paper_texts = [f"{p.title}. {p.abstract or p.summary}" for p in papers]
    paper_embeddings = embedder.encode(paper_texts)

    sims = cosine_similarity(topic_embedding, paper_embeddings)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])

    if best_sim < threshold:
        return None

    best_paper = papers[best_idx]
    return SimilarWorkMatch(
        paper_title=best_paper.title,
        paper_url=best_paper.url,
        similarity=best_sim,
    )
