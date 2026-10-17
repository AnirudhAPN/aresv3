import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Epidemic Spread Modeling and Forecasting"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a graph neural network (GNN) framework that models epidemic spread over human mobility and contact networks to improve short-term case forecasting.",
        "Evaluate how spatiotemporal GNN architectures compare against traditional compartmental models (SIR/SEIR) and classical time-series baselines for outbreak prediction accuracy.",
        "Investigate the integration of mobility, demographic, and healthcare-capacity data as node and edge features to improve GNN generalization across regions.",
        "Assess the interpretability and uncertainty quantification of GNN-based epidemic forecasts to support public health decision-making.",
    ],
    "research_questions": [
        "How do graph neural networks capture the spatial and temporal dynamics of epidemic transmission compared to compartmental and statistical models?",
        "What graph construction strategies (contact networks, mobility flows, geographic adjacency) most effectively represent disease transmission pathways for GNN-based forecasting?",
        "How does incorporating temporal attention or recurrent components into GNN architectures affect multi-step-ahead forecasting accuracy during epidemic waves?",
        "Can GNN-based models provide reliable uncertainty estimates and interpretable transmission pathways useful for public health intervention planning?",
    ],
    "keywords": [
        "graph neural networks",
        "epidemic forecasting",
        "spatiotemporal modeling",
        "mobility networks",
        "disease spread prediction",
    ],
    "research_scope": (
        "This research focuses on the application of graph neural network architectures to the problem of "
        "modeling and forecasting infectious disease spread across spatially and socially connected populations. "
        "The scope includes region-level and county-level case data, human mobility and contact network "
        "construction, and spatiotemporal deep learning architectures (e.g., GCN, GAT, ST-GNN, GraphWaveNet "
        "variants) that combine graph convolution with temporal sequence modeling. The work considers both "
        "single-disease case studies such as COVID-19 and influenza-like illness, and generalizable frameworks "
        "applicable across pathogens. Excluded from scope are purely agent-based microsimulation models without "
        "a graph-learning component and non-spatial, purely mechanistic compartmental models used in isolation."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of graph attention networks (GAT) and diffusion convolution to combine mobility flow graphs with recurrent or transformer-based temporal modules for multi-step epidemic forecasting.",
        "Growing use of human mobility data (e.g., mobile phone origin-destination flows, commuting matrices) as dynamic edge weights rather than static geographic adjacency.",
        "Rising interest in hybrid physics-informed GNNs that embed compartmental (SIR/SEIR) dynamics as constraints or auxiliary losses within the graph learning framework.",
        "Expanding benchmarking of GNN forecasters against classical baselines (ARIMA, LSTM, compartmental models) primarily on COVID-19 county- and state-level datasets.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of GNN epidemic forecasters under distribution shift, such as generalizing a model trained on one pathogen or region to a novel outbreak with different transmission characteristics.",
            "based_on_papers": ["Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: principled uncertainty quantification (e.g., calibrated prediction intervals or Bayesian GNN layers) for epidemic forecasts, which is critical for risk-aware public health decision-making.",
            "based_on_papers": ["Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized, publicly documented pipelines for constructing dynamic contact/mobility graphs from raw mobility data, limiting reproducibility and cross-study comparison of graph construction choices.",
            "based_on_papers": ["Transfer Graph Neural Networks for Pandemic Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop transfer-learning and meta-learning strategies that allow GNN epidemic models trained on historical outbreaks to adapt quickly to novel pathogens with limited early-stage data.",
        "Integrate calibrated uncertainty quantification (e.g., conformal prediction or Bayesian graph neural networks) directly into forecasting pipelines used by public health agencies.",
        "Establish open, standardized benchmarks and mobility-graph construction protocols to enable reproducible comparison of GNN architectures across diseases and geographic settings.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "World Bank Open Data", "url": "https://data.worldbank.org/", "description": "Global development, economic, and finance indicators.", "verified": True},
        {"name": "JHU CSSE COVID-19 Data Repository", "url": "https://github.com/CSSEGISandData/COVID-19", "description": "Daily time series of confirmed cases, deaths, and recoveries at country and county/state level, widely used for epidemic forecasting benchmarks.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Graph Neural Network (GNN)",
            "url": "https://arxiv.org/abs/1609.02907",
            "description": "Graph convolutional network that propagates and aggregates features across nodes representing regions or individuals connected by contact or mobility edges, used to encode spatial transmission structure.",
            "verified": True,
        },
        {
            "name": "LSTM for temporal modeling",
            "url": "https://arxiv.org/abs/1909.09586",
            "description": "Long short-term memory recurrent network used to capture temporal dependencies in case-count sequences, often combined with graph convolutional layers in spatiotemporal architectures (e.g., ST-GNN, Cola-GNN).",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Root Mean Squared Error (RMSE) between predicted and observed case counts",
        "Mean Absolute Percentage Error (MAPE) for multi-step-ahead forecasts",
        "Pearson correlation coefficient between predicted and true epidemic curves",
        "Prediction interval coverage probability (PICP) for uncertainty calibration",
    ],
    "implementation_steps": [
        "Collect and preprocess regional case, mobility, and demographic time-series data, aligning temporal resolution across sources",
        "Construct static and/or dynamic graph structures using geographic adjacency and mobility flow matrices as edge weights",
        "Design a spatiotemporal GNN architecture combining graph convolution or attention layers with a temporal module (LSTM, GRU, or attention)",
        "Train the model using a sliding-window forecasting setup with held-out future time periods for validation",
        "Benchmark against compartmental (SEIR) and classical statistical (ARIMA) baselines on the same data splits",
        "Evaluate forecast accuracy, calibration, and interpretability of learned attention weights across regions and time",
    ],
    "unverified_suggestions": [
        "Explore incorporating real-time genomic surveillance data (variant lineage) as additional node features to anticipate variant-driven transmission changes.",
        "Investigate federated learning approaches to train GNN forecasters across jurisdictions without centralizing sensitive mobility or health data.",
        "Consider integrating social media or search-trend signals as auxiliary nodes/features for early outbreak signal detection.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates the use of graph neural networks by grounding them in the spatial and relational nature of epidemic transmission, unlike purely temporal baselines.",
        "The methodology combines complementary architectural components (graph convolution/attention with recurrent temporal modeling) that reflect established practice in the spatiotemporal forecasting literature.",
        "Evaluation metrics are appropriate and cover both point-forecast accuracy (RMSE, MAPE) and calibration (PICP), reflecting a mature evaluation design.",
        "The identified gaps around generalization and uncertainty quantification are well-targeted and would meaningfully advance the field if addressed.",
    ],
    "weaknesses": [
        "The proposal does not sufficiently justify why the chosen datasets (World Bank indicators, JHU CSSE case data) provide adequate coverage of the mobility and contact-network signals the methodology assumes are available; dataset-to-method fit is under-specified.",
        "There is limited discussion of computational scalability when constructing dynamic mobility graphs at fine spatial resolution over long time horizons.",
        "The plan does not address how missing or delayed case-reporting data (a common real-world issue) would affect model training and forecast reliability.",
        "Baseline comparisons are mentioned but the proposal lacks detail on hyperparameter tuning protocols to ensure fair comparison against compartmental and statistical baselines.",
    ],
    "missing_components": [
        "No explicit plan for sourcing or constructing the human mobility/contact network data required as the core graph structure.",
        "No discussion of ethical or privacy considerations around using mobility data for epidemic modeling.",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph-based epidemic spread forecasting, based on the retrieved abstract." for t in titles]
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
