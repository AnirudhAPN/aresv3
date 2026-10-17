import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Recommendation in Cold-Start Social Networks"

PLANNER_JSON = {
    "research_objectives": [
        "Develop graph neural network recommendation models that produce useful recommendations for newly joined users with few or no prior interactions in a social network",
        "Evaluate inductive GNN architectures against transductive graph embedding methods for their ability to generalize to unseen cold-start nodes without retraining",
        "Investigate how auxiliary social signals (follow graphs, shared group membership, profile attributes) can substitute for sparse interaction history in cold-start recommendation",
        "Assess recommendation quality and fairness for cold-start users relative to established users with rich interaction histories on the same platform",
    ],
    "research_questions": [
        "Do inductive GNN architectures (e.g., GraphSAGE) outperform transductive graph embedding methods for recommending to cold-start users with no prior interaction history?",
        "Which auxiliary social signals (follow relationships, group membership, profile similarity) most effectively substitute for missing interaction history in cold-start recommendation?",
        "How much does recommendation quality for cold-start users lag behind that of established users, and how does this gap change as a new user accumulates their first few interactions?",
        "Can attention-based graph models identify which existing users or communities are most informative for bootstrapping a cold-start user's initial recommendations?",
    ],
    "keywords": [
        "graph neural network recommendation",
        "cold-start recommendation",
        "social network link prediction",
        "inductive graph representation learning",
        "social recommender systems",
    ],
    "research_scope": (
        "This research covers graph neural network approaches for recommendation in social network settings where "
        "new users have little or no interaction history, focusing on inductive representation learning and the use "
        "of auxiliary social signals such as follow graphs and group membership. It includes fairness comparison "
        "between cold-start and established users. It excludes general session-based recommendation without a social "
        "graph component, pure content-based recommendation unrelated to graph structure, and advertising or ranking "
        "systems not framed as user-facing recommendation."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of inductive GNN architectures such as GraphSAGE that can generate embeddings for previously unseen nodes without retraining the full graph",
        "Growing use of graph attention mechanisms to weight the relative informativeness of different social relationships (follows, group memberships) for a given user",
        "Rising interest in combining auxiliary social signals with sparse interaction data through heterogeneous graph representations",
        "Emerging focus on fairness-aware evaluation that explicitly compares recommendation quality between cold-start and established user populations",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of inductive and transductive graph representation methods specifically isolated to the cold-start recommendation setting.",
            "based_on_papers": ["Inductive Representation Learning on Large Graphs"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantified fairness gap analysis tracking recommendation quality as a new user accumulates their first several interactions, rather than a single static cold-start-versus-warm comparison.",
            "based_on_papers": ["Graph Attention Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic ablation of which specific auxiliary social signal (follow graph, group membership, profile similarity) contributes most to cold-start recommendation quality.",
            "based_on_papers": ["Inductive Representation Learning on Large Graphs"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish standardized cold-start recommendation benchmarks that track recommendation quality across a new user's first N interactions",
        "Develop signal-ablation protocols isolating the contribution of individual auxiliary social signals to cold-start performance",
        "Study fairness-aware training objectives that explicitly narrow the quality gap between cold-start and established users",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "GraphSAGE",
            "url": "https://arxiv.org/abs/1706.02216",
            "description": "Inductive graph neural network that generates embeddings for previously unseen nodes by aggregating neighborhood features, enabling recommendation for newly joined cold-start users without full-graph retraining.",
            "verified": True,
        },
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Attention-based graph neural network used to learn adaptive weighting of a cold-start user's available social connections when few or no interactions are present.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Recall@K and NDCG@K for cold-start user recommendations",
        "Recommendation quality gap between cold-start and established users",
        "Recommendation quality trajectory over a new user's first N interactions",
        "Coverage of distinct items/communities recommended to cold-start users",
    ],
    "implementation_steps": [
        "Construct a heterogeneous social graph combining user-item interactions, follow relationships, and group membership as node and edge features",
        "Implement GraphSAGE and GAT models to generate embeddings and recommendations for both established and simulated cold-start users",
        "Simulate cold-start conditions by masking interaction history for a held-out set of users while preserving their social graph connections",
        "Ablate individual auxiliary social signals (follow graph only, group membership only, combined) to isolate their contribution to cold-start recommendation quality",
        "Track recommendation quality as simulated cold-start users are incrementally given their first few real interactions, to characterize the warm-up trajectory",
        "Evaluate fairness by comparing recommendation quality distributions between cold-start and established user populations under matched item catalogs",
    ],
    "unverified_suggestions": ["Pinterest-20 social recommendation dataset", "Epinions social trust dataset", "Douban social recommendation dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal isolates inductive-versus-transductive representation learning specifically for the cold-start setting, rather than treating cold-start as a minor footnote in a general recommendation study",
        "Tracking recommendation quality across a new user's first several interactions provides a more nuanced picture than a single static cold-start-versus-warm comparison",
        "The signal-ablation plan for auxiliary social features gives a clear, interpretable account of what actually helps cold-start users, beyond a black-box performance number",
        "Explicit fairness comparison between cold-start and established users reflects growing platform-level concern about equitable treatment of new users",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) does not typically host social-graph recommendation datasets with follow and group-membership structure, so Pinterest-20, Epinions, and Douban remain unverified suggestions.",
        "The proposal simulates cold-start users by masking interaction history for existing users, which may not fully capture the profile and social-graph sparsity that genuinely new users exhibit in practice.",
        "There is limited discussion of privacy implications of using social graph signals (follows, group membership) for recommendation, particularly for users who may not expect such data to influence what they are shown.",
    ],
    "missing_components": [
        "A discussion of how simulated cold-start conditions differ from genuinely new users with sparse or absent social graph connections.",
        "A privacy consideration for using social graph signals in recommendation, including potential opt-out mechanisms.",
    ],
    "novelty_score": 0.69,
    "quality_score": 0.77,
}

CONFIDENCE_JSON = {"internal_consistency": 0.82, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph neural network recommendation for cold-start users in social networks, based on the retrieved abstract." for t in titles]
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
