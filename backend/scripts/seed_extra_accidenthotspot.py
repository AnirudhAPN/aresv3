import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Traffic Accident Hotspot Prediction"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a graph neural network model that predicts spatio-temporal traffic accident hotspots by representing road segments and intersections as a graph with dynamic traffic and weather features",
        "Evaluate spatio-temporal GNN architectures against grid-based CNN and classical spatial statistics baselines for accident hotspot forecasting accuracy",
        "Investigate the contribution of auxiliary features (weather, road geometry, traffic volume, time-of-day) as node and edge attributes in the graph representation",
        "Assess the practical usability of predicted hotspots for proactive traffic safety interventions, including lead time and spatial resolution trade-offs",
    ],
    "research_questions": [
        "Do spatio-temporal GNNs that model the road network topology outperform grid-based deep learning baselines for accident hotspot prediction?",
        "Which auxiliary features (weather, traffic volume, road geometry) contribute most to predictive accuracy when encoded as graph node or edge attributes?",
        "How does prediction accuracy vary with the spatial resolution of the road network graph (segment-level versus intersection-level granularity)?",
        "Can models trained on one city's road network transfer to a different city's network without substantial retraining or performance loss?",
    ],
    "keywords": [
        "graph neural network traffic accident prediction",
        "traffic accident hotspot detection",
        "spatio-temporal graph learning",
        "road network graph representation",
        "urban traffic safety analytics",
    ],
    "research_scope": (
        "This research covers graph neural network approaches for predicting spatial and temporal clusters of traffic "
        "accidents on road networks represented as graphs of segments and intersections, incorporating traffic, "
        "weather, and road-geometry features. It includes comparisons against grid-based deep learning and classical "
        "spatial statistics baselines. It excludes accident causation studies based on individual crash reconstruction, "
        "autonomous-vehicle perception systems, and non-spatial accident severity classification tasks."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of spatio-temporal graph convolutional networks that jointly model road network topology and time-varying traffic conditions",
        "Increasing fusion of weather, road geometry, and traffic-volume features as node or edge attributes rather than relying on accident history alone",
        "Shift from grid-based (raster) spatial representations toward graph-based representations that better reflect actual road connectivity",
        "Rising interest in cross-city transfer learning to reduce the cost of retraining hotspot models for new urban areas",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of graph-based versus grid-based spatial representations under matched feature sets and evaluation protocols for hotspot prediction.",
            "based_on_papers": ["Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of cross-city model transfer for accident hotspot prediction, since most studies evaluate on a single city's road network.",
            "based_on_papers": ["Deep Learning for Traffic Accident Hotspot Prediction: A Survey"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: sensitivity analysis of prediction accuracy to the spatial resolution of the underlying road network graph.",
            "based_on_papers": ["Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized benchmarks comparing graph-based and grid-based spatial representations under identical feature and evaluation conditions",
        "Study cross-city transfer and few-shot adaptation techniques to reduce retraining costs for new road networks",
        "Systematically vary road network graph resolution to characterize its effect on hotspot prediction accuracy and actionable lead time",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Spatio-Temporal Graph Convolutional Network (STGCN)",
            "url": "https://arxiv.org/abs/1709.04875",
            "description": "Graph convolutional architecture that jointly captures spatial road-network structure and temporal traffic dynamics, used as the core hotspot prediction model.",
            "verified": True,
        },
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Attention-based graph neural network used to learn adaptive weighting of neighboring road segments, providing an interpretable alternative to fixed graph convolution.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Hotspot detection precision and recall at fixed spatial resolution",
        "Area under the precision-recall curve for accident occurrence",
        "Mean spatial displacement error between predicted and actual hotspots",
        "Cross-city transfer accuracy drop",
    ],
    "implementation_steps": [
        "Construct road network graphs from open street network data, with segments/intersections as nodes and connectivity as edges",
        "Aggregate historical accident records, weather data, and traffic volume onto the graph as time-varying node and edge features",
        "Implement STGCN and GAT models to predict accident likelihood per graph node over rolling time windows",
        "Implement grid-based CNN and classical spatial statistics (e.g., kernel density estimation) baselines for comparison under matched inputs",
        "Evaluate all models on held-out time periods and, separately, on a held-out city to test cross-city generalization",
        "Vary the graph's spatial resolution (segment-level versus intersection-level) and measure the resulting change in prediction accuracy",
    ],
    "unverified_suggestions": ["NYC Open Data motor vehicle collision dataset", "UK STATS19 road safety dataset", "US DOT Fatality Analysis Reporting System (FARS)"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal makes a clear, testable case for graph-based spatial representation over grid-based baselines, with a matched-feature comparison plan",
        "Including cross-city transfer evaluation addresses a practically important generalization question that is often skipped in single-city studies",
        "The spatial-resolution sensitivity analysis adds methodological rigor beyond simply reporting a single fixed-resolution result",
        "Combining weather, traffic volume, and road geometry as graph attributes reflects a realistic and comprehensive feature design for this task",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host road-network-linked accident and traffic datasets, so the actual data sources remain unverified suggestions requiring separate government open-data access.",
        "The proposal does not address class imbalance, since accident-positive road segments are typically a small minority of all segments in any given time window.",
        "There is limited discussion of how reporting biases in official accident records (e.g., underreporting of minor collisions) might affect learned hotspot patterns.",
    ],
    "missing_components": [
        "An explicit class-imbalance handling strategy (e.g., resampling or weighted loss) for the accident-versus-no-accident prediction task.",
        "A discussion of accident-reporting bias and its potential effect on model validity across different jurisdictions.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.81, "specificity": 0.79, "completeness": 0.83}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph neural network prediction of traffic accident hotspots, based on the retrieved abstract." for t in titles]
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
