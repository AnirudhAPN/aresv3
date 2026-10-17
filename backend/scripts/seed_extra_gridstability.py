import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Electricity Grid Stability and Load Forecasting"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning models that accurately forecast short-term and long-term electricity load to support grid balancing decisions.",
        "Investigate how forecasting accuracy translates into improved frequency and voltage stability margins in modern power grids.",
        "Evaluate the robustness of neural forecasting models under high penetration of intermittent renewable generation.",
        "Design model architectures that can operate in near real-time to support grid operator decision-making.",
    ],
    "research_questions": [
        "Which deep learning architectures (LSTM, GRU, Temporal Fusion Transformer, CNN-based) provide the most accurate load forecasts across multiple time horizons?",
        "How does forecasting error propagate into grid stability metrics such as frequency deviation and reserve margin shortfalls?",
        "Can deep learning models maintain forecast accuracy as renewable energy penetration and demand-side variability increase?",
        "What data preprocessing and feature engineering strategies best capture weather, calendar, and behavioral effects on electricity demand?",
    ],
    "keywords": [
        "load forecasting",
        "grid stability",
        "LSTM neural networks",
        "renewable energy integration",
        "smart grid deep learning",
    ],
    "research_scope": "This research focuses on applying deep learning techniques, particularly recurrent and attention-based neural network architectures, to the problem of electricity load forecasting and its downstream impact on power grid stability. The scope encompasses short-term (hourly to day-ahead) and medium-term (weekly) forecasting horizons, incorporating exogenous variables such as weather data, calendar effects, and renewable generation output. The work will evaluate model performance on publicly available grid and load datasets, and examine how forecast accuracy relates to operational stability indicators like frequency regulation and spinning reserve requirements, while excluding transmission-level transient stability modeling and market-pricing mechanisms.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of LSTM and GRU recurrent architectures for short-term load forecasting due to their ability to capture temporal dependencies.",
        "Increasing use of hybrid CNN-LSTM models that combine spatial feature extraction with sequential modeling for multi-site load prediction.",
        "Emergence of attention-based and transformer architectures (e.g., Temporal Fusion Transformer) for handling long forecasting horizons and multiple exogenous inputs.",
        "Rising interest in probabilistic and uncertainty-aware forecasting to support grid reserve planning under renewable variability.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of forecasting model robustness under sudden, large-scale renewable generation drop-off events (e.g., cloud cover or wind lull transients).",
            "based_on_papers": ["Short-Term Load Forecasting Using LSTM Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct coupling of load forecast uncertainty estimates into automatic generation control or frequency regulation algorithms.",
            "based_on_papers": ["Deep Learning Approaches for Electricity Demand Forecasting", "Attention-Based Models for Power Grid Time Series Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-region transfer learning studies that test whether a forecasting model trained on one grid topology generalizes to another with different demand and renewable mix characteristics.",
            "based_on_papers": ["Hybrid CNN-LSTM Models for Smart Grid Load Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop uncertainty-quantified deep learning forecasters whose confidence intervals feed directly into grid reserve and frequency control decisions.",
        "Explore transfer learning and domain adaptation techniques to reduce the data requirements for deploying forecasting models on new grid regions.",
        "Investigate real-time, low-latency model architectures suitable for edge deployment at substation-level controllers.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "UCI Machine Learning Repository", "url": "https://archive.ics.uci.edu/", "description": "Large collection of classic ML benchmark datasets across domains.", "verified": True}],
    "algorithms": [
        {
            "name": "Long Short-Term Memory (LSTM)",
            "url": "https://www.bioinf.jku.at/publications/older/2604.pdf",
            "description": "A recurrent neural network architecture with gated memory cells that captures long-range temporal dependencies, well suited to sequential electricity load data with daily and weekly seasonality.",
            "verified": True,
        },
        {
            "name": "Temporal Fusion Transformer (TFT)",
            "url": "https://arxiv.org/abs/1912.09363",
            "description": "An attention-based deep learning architecture designed for multi-horizon time series forecasting that combines recurrent layers with interpretable self-attention over static, known-future, and observed inputs such as weather and calendar features.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Mean Absolute Percentage Error (MAPE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Error (MAE)",
        "Frequency deviation from nominal grid frequency under forecast-driven dispatch",
    ],
    "implementation_steps": [
        "Collect and clean historical electricity load, weather, and calendar data at the target grid resolution",
        "Engineer features including lagged load values, temperature, humidity, and day-of-week/holiday indicators",
        "Split data into training, validation, and test sets respecting chronological order to avoid data leakage",
        "Train baseline statistical models (ARIMA) alongside LSTM and Temporal Fusion Transformer models",
        "Tune hyperparameters using the validation set and evaluate forecast accuracy on the held-out test period",
        "Translate forecast errors into grid stability indicators by simulating reserve margin and frequency response under forecast-driven dispatch",
    ],
    "unverified_suggestions": [
        "Consider incorporating real-time SCADA telemetry feeds for online model adaptation, though access and licensing were not verified.",
        "Explore federated learning across multiple utility operators to expand training data without centralizing sensitive grid data, pending further verification of feasibility.",
        "Investigate synthetic data augmentation for rare extreme-demand events, though no verified benchmark dataset for this was found.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The research questions are well-scoped and connect forecasting accuracy directly to concrete grid stability outcomes.",
        "The methodology combines a strong recurrent baseline (LSTM) with a modern attention-based architecture (TFT), enabling meaningful comparison.",
        "Evaluation metrics span both standard forecasting error measures and a grid-specific stability indicator, strengthening applied relevance.",
        "The identified gaps around uncertainty quantification and transfer learning point to genuinely underexplored and impactful research directions.",
    ],
    "weaknesses": [
        "The proposal does not fully justify why the UCI Machine Learning Repository is sufficient given it lacks grid-topology-specific frequency response data needed for the stability evaluation.",
        "There is limited discussion of computational cost and inference latency requirements for real-time deployment.",
        "The plan does not specify how renewable generation variability will be simulated or sourced as an exogenous input.",
        "Statistical significance testing across multiple forecasting horizons is not addressed in the evaluation plan.",
    ],
    "missing_components": [
        "A clear baseline comparison against simple persistence or seasonal-naive forecasting methods.",
        "An explicit data governance and privacy plan for any utility-provided operational data.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning based electricity load forecasting and grid stability, based on the retrieved abstract." for t in titles]
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
