import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Meta-Learning for Rapid Adaptation in Personalized Recommendation Systems"

PLANNER_JSON = {
    "research_objectives": [
        "Investigate how gradient-based meta-learning algorithms can enable recommender systems to adapt to new users and items from very few interactions, reducing dependence on long historical interaction logs.",
        "Evaluate the effectiveness of meta-learned initialization strategies for mitigating cold-start problems compared to traditional transfer learning and content-based baselines.",
        "Analyze the trade-off between adaptation speed (number of gradient steps or interactions required) and recommendation accuracy across meta-learning approaches applied to personalized recommendation.",
        "Assess how task construction strategies (e.g., treating each user as a meta-learning task) affect the generalization of meta-learned recommenders to unseen user populations and item catalogs.",
    ],
    "research_questions": [
        "How does model-agnostic meta-learning (MAML) and its variants improve few-shot recommendation accuracy for new users compared to matrix factorization and content-based cold-start methods?",
        "What task sampling and episodic training strategies most effectively capture user preference heterogeneity when framing recommendation as a meta-learning problem?",
        "How many interactions (shots) are required for meta-learned recommenders to reach parity with recommenders trained on users with extensive historical data?",
        "What is the computational overhead of meta-learning-based adaptation relative to simpler fine-tuning or nearest-neighbor cold-start heuristics at inference time?",
    ],
    "keywords": [
        "meta-learning",
        "few-shot recommendation",
        "cold-start personalization",
        "model-agnostic meta-learning",
        "recommender systems",
    ],
    "research_scope": "This research focuses on meta-learning techniques, particularly gradient-based approaches such as MAML and its first-order variants, applied to rapid user and item adaptation in personalized recommendation systems. It covers episodic task formulations where individual users or user cohorts are treated as meta-learning tasks, few-shot adaptation to new items and users with sparse interaction histories, and evaluation under realistic cold-start conditions. The scope excludes non-meta-learning transfer learning approaches that do not use bi-level optimization or episodic training, recommendation domains without an explicit cold-start or rapid-adaptation requirement (e.g., static batch recommendation with abundant historical data), and meta-learning applications outside the recommendation domain such as few-shot image classification, except where used for comparative methodological grounding.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of gradient-based meta-learning (MAML, Reptile, first-order approximations) reformulated for recommendation by treating each user's interaction history as a separate few-shot adaptation task.",
        "Increasing use of memory-augmented and metric-based meta-learning architectures (e.g., prototypical networks) to avoid the computational cost of second-order gradients during user-level adaptation.",
        "Rising interest in combining meta-learning with contrastive or self-supervised pretraining to produce more transferable item and user embeddings prior to episodic fine-tuning.",
        "Emergence of meta-learning approaches specifically targeting item cold-start, where new items with no interaction history are embedded via meta-learned mappings from content features.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of meta-learned recommenders under simultaneous user and item cold-start, where both sides of the interaction matrix are sparse at once.",
            "based_on_papers": ["Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: analysis of catastrophic forgetting or preference drift when meta-learned models are continually adapted to the same user over multiple sessions.",
            "based_on_papers": ["MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarking of meta-learning-based recommenders against strong non-meta-learning cold-start baselines (e.g., content-based nearest neighbors) on identical data splits with matched shot counts.",
            "based_on_papers": ["Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Designing meta-learning objectives that jointly handle user-side and item-side cold-start within a single bi-level optimization framework.",
        "Developing continual meta-adaptation methods that update user representations incrementally across sessions without retraining from the meta-initialization each time.",
        "Establishing shot-count-matched benchmarks comparing meta-learning, transfer learning, and heuristic cold-start baselines under identical evaluation protocols.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "OpenML",
            "url": "https://www.openml.org/",
            "description": "Searchable repository of ML datasets with standardized benchmarking metadata.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Model-Agnostic Meta-Learning (MAML)",
            "url": "https://arxiv.org/abs/1703.03400",
            "description": "Gradient-based meta-learning algorithm that learns a model initialization optimized so that a small number of gradient steps on a new task (here, a new user's few interactions) yields strong task-specific performance.",
            "verified": True,
        },
        {
            "name": "Matrix Factorization Collaborative Filtering",
            "url": "https://arxiv.org/abs/1205.2618",
            "description": "Classical latent-factor baseline that decomposes the user-item interaction matrix into low-dimensional embeddings, used as a non-meta-learning comparison point for evaluating adaptation speed and cold-start accuracy gains.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Recall@K",
        "Normalized Discounted Cumulative Gain (NDCG@K)",
        "Mean Absolute Error (MAE) on rating prediction",
        "Number of interactions (shots) required to reach target accuracy",
    ],
    "implementation_steps": [
        "Formulate each user's interaction history as a meta-learning task by splitting it into a support set (few labeled interactions for adaptation) and a query set (held-out interactions for evaluation).",
        "Sample a distribution of user tasks during meta-training, ensuring coverage of diverse preference profiles and interaction sparsity levels.",
        "Initialize a base recommendation model (e.g., a neural collaborative filtering network) and apply the MAML bi-level optimization loop, computing inner-loop adaptation gradients on support sets and outer-loop meta-gradients on query sets.",
        "Evaluate the meta-learned initialization by simulating new-user and new-item scenarios, adapting with k interactions (k = 1, 5, 10) and measuring query-set performance.",
        "Compare adaptation speed and accuracy against matrix factorization and non-meta fine-tuning baselines trained from scratch or from a pretrained but non-episodic initialization.",
        "Conduct ablations on inner-loop learning rate, number of adaptation gradient steps, and task batch size to characterize sensitivity of the meta-learned model.",
    ],
    "unverified_suggestions": ["MovieLens 1M/20M cold-start splits", "Amazon Product Reviews Dataset", "Learn2learn meta-learning library"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates meta-learning by grounding the objectives in the practical need for fast adaptation under sparse-interaction cold-start conditions.",
        "The methodology specifies a concrete, reproducible episodic training pipeline from task formulation through bi-level optimization to shot-count evaluation.",
        "Evaluation metrics chosen (Recall@K, NDCG@K, MAE, shots-to-target-accuracy) directly capture both ranking quality and adaptation efficiency, which is central to the research questions.",
        "The gap analysis identifies practically relevant and underexplored issues, such as joint user-item cold-start and preference drift across sessions, that extend beyond single-shot accuracy benchmarking.",
    ],
    "weaknesses": [
        "The proposal does not fully justify why the OpenML dataset repository is an adequate proxy for user-level meta-learning tasks, since evidence sufficiency and dataset fit for genuinely sparse, session-structured interaction data remain uncertain.",
        "There is limited discussion of how the second-order gradient computation in MAML scales when the base recommendation model is large or the number of sampled user tasks per meta-batch is high.",
        "The proposal does not address how task heterogeneity (users with very different preference distributions) might destabilize the shared meta-initialization.",
        "Baseline comparisons are limited to a single classical matrix factorization method, which may understate the competitive landscape relative to modern non-meta transfer learning or metric-based few-shot recommenders.",
    ],
    "missing_components": [
        "An explicit ablation plan isolating the contribution of inner-loop adaptation steps versus meta-training task diversity on final cold-start performance.",
        "A discussion of how catalog or user-base drift over time would require periodic meta-retraining and how that cost compares to continuous fine-tuning approaches.",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to meta-learning for rapid cold-start adaptation in recommender systems, based on the retrieved abstract." for t in titles]
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
