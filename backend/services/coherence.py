"""
Cross-Agent Coherence Score — ARES's answer to "does this proposal actually
hang together as ONE plan, or five independently-plausible paragraphs?"

Generic AI research assistants (Elicit, Consensus, SciSpace, and similar
single-model tools) don't have this failure mode at all: one model call
producing one block of text is trivially self-consistent. A MULTI-agent
pipeline is structurally different — each stage is generated somewhat
independently, so nothing guarantees the Methodology agent actually
addresses the gaps the Gap agent found, or that the identified gaps trace
back to the Planner's stated research questions. A proposal can score well
on every individual agent's ACS (confidence/evidence/completion) while
still not coherently connecting as a single research plan.

This is measured with embeddings, not by asking an LLM to grade its own
consistency — the same "ground it in evidence, don't trust self-report"
rule the rest of ARES follows (see services/similarity.py, the whitelist-
grounded Methodology agent, and the never-fabricate-citations rule).
"""
from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import CoherenceReport, GapOutput, MethodologyOutput, PlannerOutput
from services.embeddings import get_embedder

# Below this cosine similarity, a gap/question is treated as "not actually
# addressed" by the stage that should be building on it. Calibrated the
# same way as services/similarity.py's novelty threshold — via observed
# real score distributions, not an arbitrary round number.
COVERAGE_THRESHOLD = 0.35


def _max_similarities(sources: list[str], targets: list[str]) -> list[float]:
    """For each source string, its best (max) cosine similarity to any
    target string. Empty inputs return neutral 0.0s rather than crashing —
    a proposal missing a section shouldn't corrupt the coherence score for
    everything else."""
    if not sources or not targets:
        return [0.0 for _ in sources]
    embedder = get_embedder()
    src_emb = embedder.encode(sources)
    tgt_emb = embedder.encode(targets)
    sims = cosine_similarity(src_emb, tgt_emb)
    return [float(row.max()) for row in sims]


def compute_coherence(
    planner: PlannerOutput, gap: GapOutput, methodology: MethodologyOutput
) -> CoherenceReport:
    # 1) Gap -> Methodology coverage: for each identified gap, is there a
    #    concrete implementation step or algorithm choice that addresses
    #    it? A gap with no similar methodology element is a proposal that
    #    names a problem and then never actually plans to solve it.
    methodology_texts = list(methodology.implementation_steps) + [
        f"{a.name}: {a.description or ''}" for a in methodology.algorithms
    ]
    gap_descriptions = [g.gap_description for g in gap.gaps]
    gap_coverage_sims = _max_similarities(gap_descriptions, methodology_texts)
    uncovered_gaps = [
        gap_descriptions[i] for i, sim in enumerate(gap_coverage_sims) if sim < COVERAGE_THRESHOLD
    ]
    gap_coverage_score = (
        sum(gap_coverage_sims) / len(gap_coverage_sims) if gap_coverage_sims else 0.5
    )

    # 2) Planner -> Gap alignment: do the identified gaps actually relate
    #    back to the stated research questions, or did the Gap agent
    #    wander off-topic from what the Planner set out to investigate?
    question_alignment_sims = _max_similarities(
        planner.research_questions, gap.current_trends + gap_descriptions
    )
    question_alignment_score = (
        sum(question_alignment_sims) / len(question_alignment_sims)
        if question_alignment_sims else 0.5
    )

    overall = 0.6 * gap_coverage_score + 0.4 * question_alignment_score

    return CoherenceReport(
        overall_score=round(max(0.0, min(1.0, overall)), 3),
        gap_coverage_score=round(gap_coverage_score, 3),
        question_alignment_score=round(question_alignment_score, 3),
        uncovered_gaps=uncovered_gaps,
    )
