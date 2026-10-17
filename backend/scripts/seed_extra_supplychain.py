import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Attention Networks for Supply Chain Disruption Risk Prediction"

PLANNER_JSON = {
    "research_objectives": [
        "Model multi-tier supply chain networks as graphs and apply Graph Attention Networks (GATs) to predict disruption risk propagation across supplier tiers",
        "Evaluate the interpretability of learned GAT attention weights for identifying critical supplier nodes that drive cascading disruption risk",
        "Compare GAT-based risk prediction against GCN, GraphSAGE, and non-graph baselines (e.g., gradient boosted trees on tabular supplier features) for disruption event prediction",
        "Investigate incorporation of dynamic/temporal graph structure to capture evolving supplier relationships and time-varying risk signals",
    ],
    "research_questions": [
        "Can attention-based GNNs predict downstream disruption propagation more accurately than static graph baselines in multi-tier supply networks?",
        "Which node and edge features (financial health indicators, geographic risk exposure, lead time, single-sourcing dependency) contribute most to GAT-learned attention weights on risk-critical suppliers?",
        "How does modeling the supply chain as a dynamic/temporal graph improve early-warning prediction of disruption events compared to static snapshot graphs?",
        "How robust are GAT-based disruption risk predictions to missing or delayed lower-tier supplier data, a common limitation of real-world supply chain visibility?",
    ],
    "keywords": [
        "graph attention network supply chain",
        "supply chain disruption prediction",
        "graph neural network risk propagation",
        "multi-tier supply chain network resilience",
        "dynamic graph neural network forecasting",
    ],
    "research_scope": (
        "This research focuses on applying Graph Attention Networks and related GNN architectures to multi-tier supply "
        "chain networks for predicting disruption risk and identifying critical supplier nodes. It covers graph "
        "construction from procurement and supplier relationship data, node/edge feature engineering (financial, "
        "geographic, logistical), static and temporal/dynamic GNN modeling, and interpretability analysis of learned "
        "attention weights. The scope excludes supply chain optimization tasks unrelated to risk (e.g., pure inventory "
        "or routing optimization) and does not cover real-time operational deployment or integration with enterprise "
        "resource planning systems, which are treated as future engineering work."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing application of GNNs (GCN, GraphSAGE, GAT) to model multi-tier supplier networks for risk and disruption prediction",
        "Use of attention mechanisms to provide interpretable identification of critical or bottleneck nodes in complex supply networks",
        "Adoption of dynamic/temporal GNN architectures to capture evolving supplier relationships and risk signals over time",
        "Integration of external risk signals (news sentiment, weather events, geopolitical indicators) as node or edge features in graph-based supply chain risk models",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a rigorous benchmark comparing GAT attention-based identification of critical suppliers against classical graph centrality measures (e.g., betweenness, eigenvector centrality) using real disruption event ground truth.",
            "based_on_papers": ["Graph Attention Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: methods for handling incomplete or missing tier-2-and-beyond supplier visibility data in GNN-based risk models, a limitation common in real enterprise deployments where only tier-1 relationships are well documented.",
            "based_on_papers": ["Inductive Representation Learning on Large Graphs"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative evaluation of dynamic/temporal GAT variants for early-warning lead time (how far in advance a disruption is flagged) rather than only post-hoc classification accuracy.",
            "based_on_papers": ["Graph Attention Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized benchmarks pairing real disruption event logs with multi-tier supplier graphs to enable reproducible comparison of GNN and centrality-based critical-node identification",
        "Design inductive GNN architectures explicitly robust to missing or partially observed lower-tier supplier connectivity",
        "Extend temporal GNN models to explicitly optimize for early-warning lead time as a training objective rather than treating it only as a post-hoc evaluation metric",
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
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Graph neural network that computes learned attention coefficients over neighboring nodes, used as the primary architecture for modeling supplier risk propagation and identifying critical nodes via attention weights.",
            "verified": True,
        },
        {
            "name": "GraphSAGE",
            "url": "https://arxiv.org/abs/1706.02216",
            "description": "Inductive graph representation learning framework that aggregates sampled neighborhood features, used as a comparison baseline and for handling previously unseen supplier nodes with incomplete connectivity.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Area under the ROC curve (AUROC) for disruption event prediction",
        "Precision@k for critical supplier identification",
        "F1 score",
        "Lead-time-to-disruption (early warning horizon)",
    ],
    "implementation_steps": [
        "Construct a multi-tier supply chain graph from procurement and ERP relationship records, with suppliers as nodes and transactional/logistical relationships as edges",
        "Engineer node features (financial risk scores, geographic exposure, historical lead times) and edge features (transaction volume, contract criticality, single-sourcing flags)",
        "Label historical disruption events (delays, shutdowns, quality failures) and align them temporally with the graph snapshots",
        "Split data temporally to avoid leakage, training on earlier periods and evaluating on later disruption events",
        "Train GAT, GraphSAGE, and GCN models alongside a gradient-boosted tree baseline on tabular supplier features",
        "Evaluate attention-weight interpretability against known critical suppliers and test robustness to simulated missing tier-2+ node data via ablation",
    ],
    "unverified_suggestions": ["SCOR-model supply chain benchmark data", "Resilinc/Everstream disruption event feeds", "MIT Supply Chain Resilience research datasets"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly frames disruption risk prediction as a graph learning problem with a concrete, interpretable output (attention-weighted critical supplier identification)",
        "Comparing GAT against both graph baselines (GCN, GraphSAGE) and a non-graph tabular baseline gives a well-rounded evaluation of when graph structure actually helps",
        "The temporal train/test split explicitly guards against label leakage, which is a common pitfall in disruption prediction studies",
        "Including lead-time-to-disruption as a metric captures the practical early-warning value of the model beyond static classification accuracy",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how ground-truth disruption labels will be sourced or validated, which is a critical dependency given that reliable, timestamped disruption event data is scarce and often proprietary",
        "Dataset fit is a concern: the whitelisted repository (OpenML) is unlikely to contain real multi-tier supplier relationship graphs, making the unverified proprietary data sources central to feasibility",
        "The plan does not address how confounding macroeconomic shocks (affecting many suppliers simultaneously) will be distinguished from genuinely graph-structure-driven cascading risk in the evaluation",
    ],
    "missing_components": [
        "A concrete data governance and privacy plan for using proprietary or semi-public supplier relationship data",
        "An ablation isolating the marginal contribution of graph structure versus node-level features alone",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.85, "specificity": 0.82, "completeness": 0.88}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph attention networks for supply chain disruption risk prediction, based on the retrieved abstract." for t in titles]
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
