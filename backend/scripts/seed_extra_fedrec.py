import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Federated Learning for Personalized On-Device Recommendation Systems"

PLANNER_JSON = {
    "research_objectives": [
        "Design a federated learning framework for training personalized recommendation models across user devices without centralizing raw interaction data",
        "Evaluate communication-efficient federated optimization algorithms (FedAvg, FedProx) for convergence and recommendation quality of on-device recommender models",
        "Investigate personalization strategies, including local fine-tuning and meta-learning adaptation, to balance global model quality with per-user personalization under federated constraints",
        "Assess the privacy-utility tradeoff of applying differential privacy noise to federated recommendation gradient updates",
    ],
    "research_questions": [
        "Can federated learning achieve recommendation accuracy competitive with centralized training while keeping raw interaction data on-device?",
        "Which federated optimization algorithm (FedAvg, FedProx, or personalized FL variants) best handles the non-IID user interaction data characteristic of recommendation settings?",
        "How much does differential privacy noise degrade recommendation quality, and what privacy budget provides an acceptable utility tradeoff?",
        "Do local fine-tuning or meta-learning based personalization layers improve per-user recommendation relevance over a single shared global federated model?",
    ],
    "keywords": [
        "federated learning recommendation system",
        "on-device personalization",
        "federated averaging non-IID data",
        "differential privacy federated learning",
        "personalized federated recommendation",
    ],
    "research_scope": (
        "This research focuses on federated learning approaches for training personalized recommendation models "
        "directly on user devices, covering federated optimization algorithms (FedAvg, FedProx), personalization "
        "mechanisms (local fine-tuning, meta-learning adapters), and differential privacy techniques applied to "
        "federated gradient updates. It uses standard recommendation benchmarks partitioned to simulate non-IID "
        "on-device data distributions. The scope excludes cross-silo federated learning between organizations, "
        "recommendation explainability, and federated learning system-level engineering concerns such as device "
        "scheduling and network protocol design, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Adoption of FedAvg and its variants (FedProx, SCAFFOLD) for training collaborative filtering and deep recommender models across distributed user devices",
        "Growing interest in personalized federated learning that maintains per-user local model layers combined with a shared global backbone",
        "Use of differential privacy mechanisms to bound information leakage from federated gradient updates in recommendation settings",
        "Exploration of communication-efficient techniques (model compression, quantization, sparse updates) to make on-device federated recommendation feasible under bandwidth constraints",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of personalized federated learning architectures (e.g., per-user embedding layers versus meta-learning adaptation) specifically for cold-start users with very few local interactions.",
            "based_on_papers": ["Federated Learning: Strategies for Improving Communication Efficiency"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantified privacy-utility tradeoff curves for differentially private federated recommendation across a range of privacy budgets on standard recommendation benchmarks.",
            "based_on_papers": ["Deep Learning with Differential Privacy"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of federated recommendation models to highly skewed, real-world-like non-IID user interaction distributions, as opposed to synthetic non-IID partitions commonly used in benchmark studies.",
            "based_on_papers": ["Federated Optimization in Heterogeneous Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop cold-start-aware personalized federated learning architectures evaluated explicitly on users with minimal interaction history",
        "Establish standardized privacy-utility benchmark curves across varying epsilon budgets for federated recommendation tasks",
        "Design federated recommendation benchmarks using realistic, empirically derived non-IID user interaction distributions rather than synthetic partitioning schemes",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "OpenML",
            "url": "https://www.openml.org/",
            "description": "Searchable repository including relevant benchmark datasets with standardized metadata.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Federated Averaging (FedAvg)",
            "url": "https://arxiv.org/abs/1602.05629",
            "description": "Foundational federated optimization algorithm that averages locally trained model updates across devices, used as the baseline aggregation method for training the global recommendation model.",
            "verified": True,
        },
        {
            "name": "FedProx",
            "url": "https://arxiv.org/abs/1812.06127",
            "description": "Federated optimization method that adds a proximal term to local objectives to improve convergence stability under non-IID and heterogeneous client data, used as a comparison against FedAvg.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Recall@K",
        "Normalized Discounted Cumulative Gain (NDCG@K)",
        "Communication rounds to convergence",
        "Privacy budget (epsilon) versus accuracy tradeoff",
    ],
    "implementation_steps": [
        "Partition a recommendation dataset (e.g., MovieLens) into non-IID per-user/per-device shards to simulate realistic on-device data distributions",
        "Implement a local on-device recommendation model (matrix factorization or neural collaborative filtering) with a local update procedure",
        "Train a global model using FedAvg and FedProx aggregation across simulated federated rounds",
        "Add a personalization layer via local fine-tuning or meta-learning adaptation on top of the shared global model",
        "Integrate differential privacy noise into gradient updates at varying epsilon budgets",
        "Evaluate Recall@K, NDCG@K, and communication cost against a centralized training baseline, and construct privacy-utility tradeoff curves",
    ],
    "unverified_suggestions": ["MovieLens 1M/20M", "Amazon Product Reviews dataset", "LEAF federated learning benchmark"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly separates the federated optimization question from the personalization question, allowing each to be evaluated with targeted ablations",
        "Including both FedAvg and FedProx as aggregation baselines directly addresses the well-known challenge of non-IID client data in federated recommendation",
        "The privacy-utility tradeoff evaluation across varying epsilon budgets gives a quantitative rather than purely qualitative treatment of the privacy dimension",
        "Communication rounds to convergence is included as a metric, reflecting practical bandwidth constraints relevant to real on-device deployment",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how simulated federated rounds will model realistic device availability and dropout patterns, which materially affect convergence in real deployments",
        "Dataset fit is a concern: MovieLens and Amazon Reviews are not natively formatted as per-device federated shards, so the non-IID partitioning scheme itself becomes a significant, underspecified modeling choice",
        "The plan does not address how the evaluation will distinguish personalization gains from federated optimization gains, since both local fine-tuning and improved aggregation could independently explain accuracy improvements",
    ],
    "missing_components": [
        "An ablation isolating the individual contribution of personalization layers versus the choice of federated aggregation algorithm",
        "A model of realistic client availability, dropout, and staleness in the federated training simulation",
    ],
    "novelty_score": 0.6,
    "quality_score": 0.74,
}

CONFIDENCE_JSON = {"internal_consistency": 0.79, "specificity": 0.77, "completeness": 0.83}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to federated learning for personalized on-device recommendation systems, based on the retrieved abstract." for t in titles]
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
