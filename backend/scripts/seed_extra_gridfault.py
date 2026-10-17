import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Power Grid Fault Localization and Diagnosis"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a graph neural network (GNN) model that localizes and classifies faults on power grid topologies using sensor (PMU/SCADA) measurements represented as graph-structured data",
        "Evaluate message-passing GNN architectures for their ability to exploit grid bus-branch topology versus topology-agnostic deep learning baselines for fault localization accuracy",
        "Investigate the robustness of GNN-based fault diagnosis to missing or noisy sensor readings and to topology changes such as switching operations and N-1 contingencies",
        "Assess the interpretability of learned GNN attention or message weights for identifying likely fault locations in a way usable by grid operators",
    ],
    "research_questions": [
        "Do GNNs that explicitly encode grid bus-branch topology outperform topology-agnostic deep learning baselines for fault localization accuracy?",
        "How robust is GNN-based fault diagnosis to partial sensor observability and to measurement noise in PMU/SCADA data?",
        "Can GNN-based fault localization models generalize to topology changes, such as line switching or N-1 contingency configurations, without retraining?",
        "Which interpretability techniques most faithfully highlight the true fault location to a human grid operator from learned GNN attention or message weights?",
    ],
    "keywords": [
        "graph neural network fault diagnosis",
        "power grid fault localization",
        "PMU phasor measurement fault detection",
        "smart grid anomaly detection",
        "topology-aware fault classification",
    ],
    "research_scope": (
        "This research covers GNN-based approaches for fault localization and classification on transmission "
        "and distribution power grids using PMU or SCADA sensor data represented over the grid's bus-branch "
        "topology graph, including robustness to partial observability and topology changes. The scope excludes "
        "load forecasting, generation scheduling and dispatch optimization, and non-graph-based fault detection "
        "methods except where used as comparison baselines."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Adoption of graph convolutional and graph attention networks that directly encode bus-branch topology for fault localization",
        "Fusion of PMU (high-frequency phasor) and SCADA (lower-frequency) measurements as heterogeneous node and edge features in GNN pipelines",
        "Growing interest in robustness testing under partial sensor observability, reflecting real-world sparse PMU deployment",
        "Use of GNN explainability methods, including attention visualization and subgraph relevance extraction, to support operator trust and regulatory auditability",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of GNN fault localization accuracy versus topology-agnostic deep baselines under matched sensor input conditions.",
            "based_on_papers": ["Graph Neural Networks for Fault Location in Power Distribution Systems"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of robustness to combined sensor noise and partial observability simultaneously, since most studies test these factors independently.",
            "based_on_papers": ["Physics-Informed Graph Neural Networks for Power Grid Fault Diagnosis"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: validation of GNN interpretability outputs against ground-truth operator diagnostic reasoning or historical fault records.",
            "based_on_papers": ["Graph Neural Networks for Fault Location in Power Distribution Systems"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop combined noise-plus-partial-observability robustness benchmarks reflecting realistic PMU/SCADA deployment gaps",
        "Validate GNN explainability outputs against historical utility fault-report data to assess operator trust and usability",
        "Study transfer and generalization of trained GNN fault localizers across grid topologies of different sizes and voltage levels",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Graph Convolutional Network (GCN)",
            "url": "https://arxiv.org/abs/1609.02907",
            "description": "Spectral-approximation graph neural network used to propagate sensor measurement information across the grid bus-branch topology for fault localization.",
            "verified": True,
        },
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Attention-based graph neural network that learns to weight neighboring bus/branch contributions, used both for fault classification and as an interpretability signal for operators.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Fault localization accuracy (correct bus/branch identification)",
        "Fault-type classification F1-score",
        "Localization error distance (number of buses or kilometers from true fault)",
        "Robustness accuracy under simulated missing-sensor conditions",
    ],
    "implementation_steps": [
        "Construct grid topology graphs (bus-branch structure) from standard IEEE test system definitions, with PMU/SCADA sensor placements as node/edge features",
        "Simulate fault scenarios (type, location, severity) using power system simulation tools to generate labeled training data",
        "Implement and train GCN and GAT models to predict fault location and type from simulated sensor measurements over the topology graph",
        "Implement topology-agnostic deep learning baselines (MLP, CNN over flattened sensor arrays) for comparison under matched input conditions",
        "Evaluate robustness by systematically masking sensor readings and injecting measurement noise, and by testing on topology variants (line switching, N-1 contingencies)",
        "Apply GNN explainability techniques (attention weight visualization, subgraph relevance extraction) and assess alignment with known fault locations",
    ],
    "unverified_suggestions": ["IEEE 118-bus test system", "IEEE 33-bus distribution test feeder", "PSML (Power System Machine Learning) dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal makes a clear, testable case for topology-aware GNN modeling over topology-agnostic baselines, with a concrete comparison plan",
        "Robustness evaluation under sensor noise, partial observability, and topology change is treated as a first-class objective rather than an afterthought",
        "The interpretability component directly targets operator usability, an important practical requirement for grid fault diagnosis adoption",
        "Simulation-based data generation using standard IEEE test systems provides a reproducible and widely recognized experimental foundation",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal relies primarily on simulated fault data, and it does not address how well simulation-trained models would transfer to real utility SCADA/PMU recordings with their own idiosyncratic noise characteristics",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host power system simulation benchmarks such as IEEE test feeders, so those remain unverified suggestions requiring separate simulation setup",
        "The plan does not address multi-fault or cascading-fault scenarios, which are operationally important but more complex than the single-fault localization setting implied by the current design",
    ],
    "missing_components": [
        "A sim-to-real validation step comparing simulation-trained model performance against real (or realistically noised) utility measurement data",
        "An explicit extension or discussion of multi-fault and cascading-fault localization beyond the single-fault case",
    ],
    "novelty_score": 0.65,
    "quality_score": 0.74,
}

CONFIDENCE_JSON = {"internal_consistency": 0.79, "specificity": 0.77, "completeness": 0.82}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph neural network fault localization on power grids, based on the retrieved abstract." for t in titles]
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
