import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Convolutional Networks for Recommender Systems in E-Commerce"

PLANNER_JSON = {
    "research_objectives": [
        "Investigate how graph convolutional networks (GCNs) can model high-order user-item interactions in e-commerce recommendation to improve accuracy over traditional collaborative filtering.",
        "Evaluate the effectiveness of GCN-based embedding propagation for mitigating cold-start and data sparsity problems in large-scale e-commerce catalogs.",
        "Analyze the scalability and inference latency trade-offs of GCN-based recommenders when deployed on production-scale e-commerce interaction graphs.",
        "Assess how auxiliary signals such as product category graphs and co-purchase relations can be integrated into GCN architectures to boost recommendation relevance.",
    ],
    "research_questions": [
        "How does explicit modeling of multi-hop user-item connectivity via GCNs improve top-N recommendation accuracy compared to matrix factorization baselines?",
        "What GCN architectural choices (number of layers, aggregation function, normalization) most affect performance on sparse e-commerce interaction data?",
        "Can graph-based embedding propagation reduce cold-start degradation for newly listed products or new users?",
        "What is the computational overhead of GCN-based recommenders relative to simpler embedding-based models at e-commerce scale?",
    ],
    "keywords": [
        "graph convolutional networks",
        "recommender systems",
        "e-commerce personalization",
        "collaborative filtering",
        "graph neural networks",
    ],
    "research_scope": "This research focuses on graph convolutional network architectures applied specifically to implicit-feedback recommendation tasks within e-commerce settings, such as product click, add-to-cart, and purchase interactions. It covers embedding propagation methods, neighborhood aggregation strategies, and scalability considerations for bipartite user-item graphs and heterogeneous graphs incorporating item attributes. The scope excludes session-based sequential recommenders that do not rely on graph structures, non-graph deep learning recommenders (e.g., pure autoencoder or transformer-based models without graph convolution), and recommendation domains outside e-commerce such as social media content feeds or video streaming, except where used for comparative benchmarking.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of light-weight graph convolution operators (e.g., removing feature transformation and nonlinearity) to improve training efficiency on large bipartite user-item graphs.",
        "Growing interest in combining GCN-based collaborative signals with heterogeneous side information such as product category graphs, knowledge graphs, and co-purchase networks.",
        "Shift toward scalable graph sampling and mini-batch training techniques to make GCN recommenders feasible for industrial-scale e-commerce catalogs with millions of items.",
        "Emergence of self-supervised and contrastive learning objectives layered on top of GCN architectures to improve robustness to interaction sparsity and popularity bias.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of GCN-based recommenders under real-world concept drift, where product catalogs and user preferences shift rapidly over time in e-commerce settings.",
            "based_on_papers": ["Neural Graph Collaborative Filtering", "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: fine-grained analysis of fairness and exposure bias introduced by GCN embedding propagation toward popular versus long-tail products.",
            "based_on_papers": ["Neural Graph Collaborative Filtering"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarking of GCN recommenders against transformer-based sequential recommenders on identical e-commerce datasets with matched preprocessing.",
            "based_on_papers": ["LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Developing continual-learning variants of GCN recommenders that efficiently update embeddings as new products and interactions arrive without full graph retraining.",
        "Designing fairness-aware graph convolution objectives that explicitly regularize exposure disparity across product popularity tiers.",
        "Creating unified benchmarks that directly compare graph-based and sequence-based recommenders under identical e-commerce data splits and evaluation protocols.",
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
            "name": "Graph Neural Network (GNN)",
            "url": "https://arxiv.org/abs/1609.02907",
            "description": "Semi-supervised graph convolutional network architecture that propagates and aggregates neighbor embeddings across the user-item interaction graph to learn representations capturing high-order connectivity patterns.",
            "verified": True,
        },
        {
            "name": "Matrix Factorization Collaborative Filtering",
            "url": "https://arxiv.org/abs/1205.2618",
            "description": "Classical latent-factor baseline that decomposes the user-item interaction matrix into low-dimensional embeddings, used as a non-graph comparison point for evaluating GCN gains.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Recall@K",
        "Normalized Discounted Cumulative Gain (NDCG@K)",
        "Hit Ratio@K",
        "Mean Reciprocal Rank (MRR)",
    ],
    "implementation_steps": [
        "Construct a bipartite user-item interaction graph from e-commerce clickstream and transaction logs, filtering out users and items with insufficient interaction counts.",
        "Preprocess and normalize the adjacency matrix using symmetric degree normalization to stabilize embedding propagation across graph layers.",
        "Initialize user and item embedding tables and define the multi-layer graph convolution aggregation function, including the choice of layer combination (e.g., weighted sum or concatenation).",
        "Train the model using a pairwise ranking loss such as Bayesian Personalized Ranking (BPR) with negative sampling over the interaction graph.",
        "Tune hyperparameters including embedding dimension, number of propagation layers, learning rate, and regularization strength via validation set performance.",
        "Evaluate the trained model on held-out interactions using top-N ranking metrics and compare against non-graph collaborative filtering baselines.",
    ],
    "unverified_suggestions": ["Amazon Product Co-Purchasing Network Dataset", "Taobao User Behavior Dataset", "DGL-KE"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates the use of graph convolutional networks by grounding the objectives in the high-order connectivity structure inherent to e-commerce user-item interactions.",
        "The methodology specifies a concrete, reproducible pipeline from graph construction through embedding propagation to ranking evaluation.",
        "Evaluation metrics chosen (Recall@K, NDCG@K, Hit Ratio@K, MRR) are standard and appropriate for top-N implicit-feedback recommendation tasks.",
        "The gap analysis identifies practically relevant and underexplored issues, such as concept drift and exposure fairness, that extend beyond simple accuracy benchmarking.",
    ],
    "weaknesses": [
        "The proposal does not fully justify why the OpenML dataset repository is an adequate proxy for large-scale e-commerce interaction data, since evidence sufficiency and dataset fit for genuinely sparse, high-volume catalog settings remain uncertain.",
        "There is limited discussion of how cold-start users or items without any prior interaction edges would be handled within the graph propagation framework.",
        "The proposal does not address computational scalability trade-offs in enough depth given that industrial e-commerce graphs can contain hundreds of millions of edges.",
        "Baseline comparisons are limited to a single classical matrix factorization method, which may understate the competitive landscape relative to modern sequential or hybrid recommenders.",
    ],
    "missing_components": [
        "An explicit ablation plan isolating the contribution of graph depth (number of convolution layers) versus embedding dimensionality on final performance.",
        "A discussion of privacy and data governance considerations when constructing interaction graphs from real user clickstream data.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph-based recommendation in e-commerce, based on the retrieved abstract." for t in titles]
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
