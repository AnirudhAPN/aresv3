import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Contrastive Learning for Cross-Lingual Document Retrieval in Low-Resource Languages"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a contrastive learning framework that aligns document representations across a high-resource pivot language and a low-resource target language for cross-lingual retrieval",
        "Evaluate the effect of parallel versus weakly-aligned comparable corpora as contrastive training signal on cross-lingual embedding quality for low-resource languages",
        "Investigate multilingual pretrained encoder backbones (e.g., mBERT, XLM-R, LaBSE) as initialization points for contrastive fine-tuning in low-resource settings",
        "Assess retrieval performance degradation as the amount of available parallel or comparable training data for the low-resource language decreases",
    ],
    "research_questions": [
        "How does contrastive fine-tuning on comparable, non-parallel corpora compare to fine-tuning on true parallel corpora for cross-lingual retrieval quality in low-resource languages?",
        "Which multilingual pretrained encoder backbones provide the best initialization for contrastive cross-lingual retrieval fine-tuning under limited target-language data?",
        "What is the minimum amount of parallel or comparable data needed to achieve acceptable cross-lingual retrieval accuracy for a genuinely low-resource language?",
        "Do hard-negative mining strategies designed for high-resource contrastive retrieval transfer effectively to low-resource cross-lingual settings?",
    ],
    "keywords": [
        "cross-lingual document retrieval",
        "contrastive learning low-resource language",
        "multilingual sentence embeddings",
        "cross-lingual information retrieval",
        "comparable corpora alignment",
    ],
    "research_scope": (
        "This research covers contrastive representation learning methods for cross-lingual document and "
        "sentence retrieval between a high-resource pivot language and a low-resource target language, "
        "including data-efficiency analysis and comparison of multilingual encoder backbones. The scope "
        "excludes machine translation quality evaluation as a standalone objective, purely monolingual "
        "retrieval, and speech-based cross-lingual retrieval."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Fine-tuning multilingual pretrained encoders (mBERT, XLM-R, LaBSE) with contrastive objectives such as InfoNCE for cross-lingual sentence and document alignment",
        "Increasing use of mined or weakly-aligned comparable corpora, such as Wikipedia and CommonCrawl-derived data, as substitutes for scarce true parallel data in low-resource languages",
        "Adoption of hard-negative mining and in-batch negative sampling strategies originally developed for dense passage retrieval, adapted to cross-lingual settings",
        "Growing interest in language-family transfer approaches that bootstrap low-resource retrieval performance from related higher-resource languages",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of parallel versus comparable-corpus contrastive training under matched data volume for genuinely low-resource languages.",
            "based_on_papers": ["Language-agnostic BERT Sentence Embedding", "Unsupervised Cross-lingual Representation Learning at Scale"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: data-efficiency curves quantifying retrieval accuracy as a function of available parallel or comparable sentence pairs for low-resource target languages.",
            "based_on_papers": ["Language-agnostic BERT Sentence Embedding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: the transferability of hard-negative mining strategies developed for high-resource dense retrieval to the noisier alignment signal available in low-resource cross-lingual contrastive training.",
            "based_on_papers": ["Dense Passage Retrieval for Open-Domain Question Answering"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Construct standardized low-resource data-efficiency benchmarks spanning multiple language families and data volumes",
        "Study language-family transfer bootstrapping, using related higher-resource languages to initialize low-resource contrastive training",
        "Develop hard-negative mining techniques specifically adapted to the noisier alignment signal available in low-resource comparable corpora",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "LaBSE (Language-agnostic BERT Sentence Embedding)",
            "url": "https://arxiv.org/abs/2007.01852",
            "description": "Multilingual dual-encoder model producing language-agnostic sentence embeddings, used as a pretrained initialization for contrastive cross-lingual retrieval fine-tuning.",
            "verified": True,
        },
        {
            "name": "Dense Passage Retrieval (contrastive InfoNCE-style training)",
            "url": "https://arxiv.org/abs/2004.04906",
            "description": "Dual-encoder contrastive retrieval framework with in-batch and hard-negative sampling, adapted here to cross-lingual document retrieval between pivot and low-resource languages.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Mean Reciprocal Rank (MRR)",
        "Recall@K",
        "Mean Average Precision (mAP)",
        "Cross-lingual retrieval accuracy at rank 1",
    ],
    "implementation_steps": [
        "Assemble parallel and comparable corpora between the high-resource pivot language and the selected low-resource target language, subsampled to controlled data-volume tiers",
        "Initialize dual encoders from a multilingual pretrained backbone (mBERT, XLM-R, or LaBSE) for contrastive fine-tuning",
        "Implement contrastive fine-tuning with in-batch negatives and hard-negative mining, training separate variants on parallel-only versus comparable-only corpora",
        "Evaluate cross-lingual document retrieval performance on held-out query-document pairs across the controlled data-volume tiers",
        "Compare retrieval accuracy across the three multilingual encoder backbones under matched fine-tuning conditions",
        "Analyze the data-efficiency curve by plotting retrieval metrics against training data volume to identify minimum viable data thresholds",
    ],
    "unverified_suggestions": ["OPUS parallel corpora", "Tatoeba", "CC-Aligned (CommonCrawl-aligned comparable corpora)"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal isolates a specific, testable comparison (parallel versus comparable-corpus training) that directly addresses a known practical constraint in low-resource NLP",
        "The data-efficiency curve methodology provides an interpretable, actionable output for practitioners deciding how much data to collect for a new low-resource language",
        "Comparing three distinct multilingual encoder backbones under matched conditions strengthens the generalizability of the conclusions",
        "The evaluation metrics (MRR, Recall@K, mAP) are standard and appropriate for retrieval quality assessment",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify which low-resource language(s) will be selected as the target, and the choice could substantially affect results depending on script, morphology, and available comparable corpus size",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host parallel or comparable multilingual text corpora such as OPUS or CC-Aligned, so those remain unverified suggestions requiring separate access",
        "The plan does not address how translation or alignment noise inherent in comparable (as opposed to parallel) corpora will be quantified or filtered before use as training signal",
    ],
    "missing_components": [
        "An explicit criterion or rationale for selecting the target low-resource language(s) to ensure results are not idiosyncratic to one language",
        "A noise-filtering or quality-assessment step for comparable corpus alignment pairs prior to contrastive training",
    ],
    "novelty_score": 0.71,
    "quality_score": 0.77,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to contrastive cross-lingual retrieval for low-resource languages, based on the retrieved abstract." for t in titles]
        return json.dumps({"summaries": summaries})
    if "Gap Analysis Agent" in prompt:
        return json.dumps(GAP_JSON)
    if "Methodology Agent" in prompt:
        return json.dumps(METHODOLOGY_JSON)
    raise AssertionError(f"Unexpected generate() prompt: {prompt[:200]}")


def mock_judge(prompt, temperature=0.0, agent=None, **kwargs):
    if "internal_consistency" in prompt:
        return json.dumps(CONFIDENCE_JSON)
    if "qualitative critique" in prompt or "Reviewer Agent" in prompt:
        return json.dumps(REVIEWER_JSON)
    raise AssertionError(f"Unexpected judge() prompt: {prompt[:200]}")


def main():
    from services.memory import memory_service
    from services.llm import llm_service
    import orchestrator as orch

    session_id = memory_service.create_session(TOPIC, source="seed")
    print("session_id:", session_id)

    with patch.object(llm_service, "generate", side_effect=mock_generate), \
         patch.object(llm_service, "judge", side_effect=mock_judge):
        final_state = orch.run_pipeline(session_id, TOPIC, condition="adaptive")

    print("completed_agents:", final_state.get("completed_agents"))
    print("acs_scores:", json.dumps(final_state.get("acs_scores", {}), indent=2))
    print("flags:", final_state.get("flags"))
    session = memory_service.get_session(session_id)
    print("FINAL STATUS:", session["status"])


if __name__ == "__main__":
    main()
