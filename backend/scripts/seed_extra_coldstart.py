import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer-Based Recommendation for Cold-Start Users in Streaming Platforms"

PLANNER_JSON = {
    "research_objectives": [
        "Design a transformer-based sequential recommendation model that produces high-quality recommendations for users with little to no interaction history on streaming platforms",
        "Evaluate the use of side information (content metadata, onboarding survey responses, first-session behavior) fused into transformer architectures to compensate for sparse interaction sequences",
        "Investigate meta-learning and few-shot adaptation techniques applied to transformer recommenders to accelerate personalization for newly onboarded users",
        "Compare cold-start transformer recommenders against collaborative filtering and hybrid baselines on ranking quality and recommendation diversity for new users",
    ],
    "research_questions": [
        "How much do content-based side features improve transformer sequential recommendation accuracy for users with fewer than a defined threshold of historical interactions?",
        "Can meta-learning adaptation applied to transformer recommenders reduce the number of interactions needed to reach acceptable recommendation quality for a new user?",
        "Does attention over onboarding signals (explicit preference surveys, first-session behavior) meaningfully improve early-session recommendation relevance compared to relying on implicit interactions alone?",
        "What is the tradeoff between recommendation accuracy and catalog diversity/serendipity for cold-start users under different transformer architectures?",
    ],
    "keywords": [
        "cold-start recommendation",
        "sequential recommendation transformer",
        "session-based recommender systems",
        "meta-learning recommendation",
        "content-based recommendation streaming",
    ],
    "research_scope": (
        "This research focuses on transformer-based sequential and session recommendation models specifically "
        "targeting cold-start users, defined as new users or those with very short interaction histories, on "
        "streaming video or music platforms. It covers side-information fusion (content metadata, onboarding "
        "signals) and meta-learning adaptation strategies applied on top of transformer backbones. The scope "
        "excludes cold-start item recommendation, treats non-transformer collaborative filtering only as a "
        "baseline comparison, and excludes cross-platform transfer learning between unrelated recommendation "
        "domains."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Adoption of self-attention sequential recommenders (SASRec, BERT4Rec) as backbones later adapted for cold-start scenarios",
        "Fusion of content and metadata embeddings (genre, cast, audio features) into transformer input to offset sparse interaction history",
        "Meta-learning approaches (e.g., MeLU) applied to recommendation for fast adaptation to new users with minimal interactions",
        "Use of large pretrained language or embedding models to encode item descriptions for zero-shot or near-zero-shot cold-start recommendation",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a direct comparison of meta-learning adaptation versus simple side-feature fusion for transformer-based cold-start recommendation under a shared evaluation protocol.",
            "based_on_papers": ["Self-Attentive Sequential Recommendation", "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of recommendation diversity and catalog coverage specifically for cold-start users, since most benchmarks report only ranking accuracy metrics.",
            "based_on_papers": ["Self-Attentive Sequential Recommendation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: generalization of meta-learning cold-start recommenders across streaming domains (video versus music) that differ in content metadata richness.",
            "based_on_papers": ["MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design unified cold-start benchmarks that jointly report accuracy, diversity, and catalog coverage across interaction-history buckets",
        "Explore architectures that combine meta-learning adaptation with content-embedding fusion within a single transformer model",
        "Study cross-domain transfer of cold-start recommenders between video and music streaming catalogs with differing metadata richness",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Self-Attentive Sequential Recommendation (SASRec)",
            "url": "https://arxiv.org/abs/1808.09781",
            "description": "Transformer-based sequential recommender using unidirectional self-attention over interaction history, serving as the core backbone adapted here for cold-start side-feature fusion.",
            "verified": True,
        },
        {
            "name": "BERT4Rec",
            "url": "https://arxiv.org/abs/1904.06690",
            "description": "Bidirectional transformer sequential recommender trained with a masked item prediction objective, used as a comparison architecture for cold-start adaptation experiments.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Hit Rate@K",
        "Normalized Discounted Cumulative Gain (NDCG@K)",
        "Mean Reciprocal Rank (MRR)",
        "Catalog coverage / recommendation diversity",
    ],
    "implementation_steps": [
        "Assemble a streaming interaction dataset with timestamps, item metadata, and, where available, onboarding survey responses",
        "Construct cold-start evaluation splits by truncating user histories to simulate varying degrees of interaction sparsity (e.g., 0, 1, 5 prior interactions)",
        "Implement a transformer sequential recommender baseline (SASRec/BERT4Rec) and augment it with fused content-metadata and onboarding-signal embeddings",
        "Implement a meta-learning adaptation layer (e.g., gradient-based fast adaptation) on top of the transformer backbone for rapid personalization to new users",
        "Train and evaluate all variants on the cold-start splits, computing Hit Rate@K, NDCG@K, MRR, and diversity metrics per interaction-history bucket",
        "Perform ablations isolating the contribution of side-feature fusion versus meta-learning adaptation to identify which mechanism drives cold-start gains",
    ],
    "unverified_suggestions": ["MovieLens-25M", "Netflix Prize dataset", "Last.fm dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal targets a practically important and well-defined problem (cold-start personalization) with clear, measurable research questions",
        "The methodology isolates two candidate mechanisms (side-feature fusion and meta-learning adaptation) with a planned ablation to distinguish their individual contributions",
        "Evaluation explicitly includes diversity and catalog coverage alongside accuracy, addressing a common blind spot in cold-start recommendation studies",
        "Cold-start evaluation splits based on truncated interaction history provide a controlled, reproducible way to study the sparsity regime",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how onboarding survey data would be obtained or standardized, since such data is rarely available in public benchmark datasets",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) is not a typical source for streaming interaction datasets such as MovieLens or Last.fm, so those remain unverified suggestions requiring separate access",
        "The plan does not address potential popularity bias, where cold-start recommenders may default to globally popular items rather than genuinely personalized ones, inflating accuracy metrics without indicating true personalization quality",
    ],
    "missing_components": [
        "A explicit popularity-bias control or debiased evaluation protocol to distinguish genuine personalization from popularity-driven accuracy",
        "A plan for statistical significance testing across interaction-history buckets and model variants",
    ],
    "novelty_score": 0.63,
    "quality_score": 0.72,
}

CONFIDENCE_JSON = {"internal_consistency": 0.78, "specificity": 0.76, "completeness": 0.80}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based cold-start recommendation for streaming platforms, based on the retrieved abstract." for t in titles]
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
