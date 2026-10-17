import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Fraud Ring Detection in Cryptocurrency Transactions"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a graph neural network (GNN) framework capable of identifying coordinated fraud rings within large-scale cryptocurrency transaction networks.",
        "Evaluate the effectiveness of graph-based representation learning versus traditional rule-based and tabular machine learning approaches for detecting illicit wallet clusters.",
        "Investigate how transaction graph topology (e.g. layering, peel chains, mixing patterns) can be encoded as features for fraud ring identification.",
        "Assess the scalability and real-time applicability of GNN-based detection pipelines on high-throughput blockchain ledgers such as Bitcoin and Ethereum.",
    ],
    "research_questions": [
        "Can graph neural networks outperform traditional feature-engineered classifiers in detecting organized fraud rings in cryptocurrency transaction graphs?",
        "Which graph structural patterns (e.g. dense subgraphs, star topologies, chain-hopping) are most indicative of coordinated fraudulent activity?",
        "How do GNN-based fraud detectors generalize across different blockchains and evolving obfuscation techniques used by fraud rings?",
        "What is the trade-off between detection accuracy and computational scalability when applying GNNs to million-node transaction graphs?",
    ],
    "keywords": [
        "graph neural networks",
        "cryptocurrency fraud detection",
        "fraud ring identification",
        "blockchain transaction graphs",
        "anomaly detection in graphs",
    ],
    "research_scope": (
        "This research focuses on applying graph neural network architectures to the problem of detecting "
        "organized fraud rings in cryptocurrency transaction networks, primarily on public blockchains such as "
        "Bitcoin and Ethereum. The scope covers graph construction from raw transaction data, node and edge "
        "feature engineering, supervised and semi-supervised GNN training on labeled illicit-address datasets, "
        "and evaluation of detection performance against non-graph baselines. It excludes off-chain fraud "
        "(e.g. exchange account takeover) and focuses specifically on on-chain, graph-structured evidence of "
        "collusive fraudulent behavior."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of graph convolutional and graph attention networks (GCN, GAT, GraphSAGE) for illicit Bitcoin address classification, largely driven by the Elliptic dataset.",
        "Growing use of temporal graph networks to capture the evolving nature of transaction flows and account behavior over time.",
        "Shift from single-node classification toward subgraph and community-detection approaches to identify entire fraud rings rather than isolated illicit addresses.",
        "Integration of heterogeneous graph representations that combine wallet addresses, smart contracts, and exchange entities into unified fraud detection models.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: robust evaluation of GNN fraud detectors under adversarial evasion, where fraud rings deliberately restructure transaction graphs (e.g. peel chains, mixing services) to evade detection.",
            "based_on_papers": ["Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-chain fraud ring detection that tracks coordinated actors moving funds between Bitcoin, Ethereum, and layer-2 bridges to obscure provenance.",
            "based_on_papers": ["Inspection-L: Self-Supervised GNN Node Embeddings for Money Laundering Detection in Bitcoin"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: interpretability of GNN fraud predictions for regulatory and law-enforcement use, where black-box subgraph classifications are insufficient for legal action without explainable evidence trails.",
            "based_on_papers": ["Elliptic Data Set: Anti-Money Laundering in Bitcoin"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop adversarially robust GNN architectures trained with simulated evasion strategies to withstand fraud ring restructuring attempts.",
        "Build cross-chain heterogeneous graph datasets and models that unify address activity across multiple blockchains and bridges.",
        "Incorporate explainability techniques (e.g. GNNExplainer, subgraph attribution) so fraud ring flags can be audited by compliance and law enforcement teams.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "Kaggle Datasets",
            "url": "https://www.kaggle.com/datasets",
            "description": "Community-contributed datasets spanning nearly every domain, with notebooks.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Graph Neural Network (GNN)",
            "url": "https://arxiv.org/abs/1609.02907",
            "description": "Graph convolutional network architecture that aggregates neighborhood transaction features to classify wallet addresses and subgraphs as licit or illicit.",
            "verified": True,
        },
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Attention-based graph model that learns to weight neighboring transactions differently, useful for highlighting high-risk edges within a suspected fraud ring.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision and recall on illicit-address classification, given severe class imbalance",
        "F1-score at both node level (illicit wallets) and subgraph/community level (fraud rings)",
        "Area under the precision-recall curve (AUPRC)",
        "Ring detection accuracy measured via community-detection overlap with ground-truth labeled fraud clusters",
    ],
    "implementation_steps": [
        "Collect and preprocess a labeled transaction graph dataset (e.g. Elliptic Bitcoin dataset) into node, edge, and feature tensors",
        "Construct the transaction graph with wallet addresses as nodes and transfers as directed, timestamped edges",
        "Engineer node features from transaction metadata (amounts, timestamps, degree, clustering coefficient) and edge features from transfer patterns",
        "Train a GNN classifier (GCN/GAT/GraphSAGE) using semi-supervised learning given the small proportion of labeled illicit nodes",
        "Apply community detection or subgraph pooling on top of learned node embeddings to isolate coordinated fraud rings rather than isolated illicit nodes",
        "Evaluate against non-graph baselines (e.g. random forest on tabular features) and report precision, recall, F1, and AUPRC across both node- and ring-level tasks",
    ],
    "unverified_suggestions": [
        "Explore temporal graph neural networks (e.g. TGN) to capture the evolution of fraud rings across transaction time windows.",
        "Investigate self-supervised pretraining on unlabeled transaction graphs to improve generalization given limited labeled fraud data.",
        "Consider federated learning across exchanges to train fraud detection models without centralizing sensitive transaction data.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly targets a well-defined and practically important problem: detecting coordinated fraud rings rather than only isolated illicit transactions.",
        "The methodology grounds itself in an established, publicly available benchmark (the Elliptic Bitcoin dataset), which supports reproducibility.",
        "The choice of GCN and GAT architectures is well justified given their demonstrated success in prior illicit-address classification literature.",
        "Evaluation metrics appropriately account for class imbalance by emphasizing precision, recall, and AUPRC rather than raw accuracy.",
    ],
    "weaknesses": [
        "The gap analysis and methodology rely on a small set of retrieved papers, so claims about unaddressed cross-chain and adversarial-evasion gaps should be treated as provisional pending a broader literature sweep.",
        "The proposal does not specify how fraud ring ground-truth labels will be constructed for datasets that only provide node-level (not ring-level) illicit labels.",
        "No baseline comparison against simpler graph heuristics (e.g. degree-based or clustering-coefficient thresholds) is included alongside the ML baselines.",
        "The plan does not address how the model would handle newly created wallet addresses with no transaction history (cold-start nodes).",
    ],
    "missing_components": [
        "A concrete data labeling or weak-supervision strategy for identifying ring-level (as opposed to node-level) ground truth.",
        "A discussion of computational cost and inference latency for deploying the GNN in a near-real-time transaction monitoring setting.",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.81, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to GNN-based fraud ring detection in cryptocurrency transaction networks, based on the retrieved abstract." for t in titles]
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
