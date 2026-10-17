import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer-Based Time Series Forecasting for Renewable Energy Generation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based architectures that model long-range temporal dependencies in solar and wind power generation time series.",
        "Quantify forecasting accuracy improvements of attention-based models over classical statistical and recurrent baselines across multiple forecast horizons.",
        "Investigate the integration of exogenous weather and irradiance covariates into transformer forecasting pipelines.",
        "Assess the computational and data efficiency trade-offs of transformer models when deployed for grid-scale renewable energy forecasting.",
    ],
    "research_questions": [
        "How do transformer architectures compare to LSTM and statistical baselines in short- and long-horizon renewable energy forecasting accuracy?",
        "What attention mechanisms best capture seasonal and diurnal periodicity in solar and wind generation data?",
        "How does incorporating exogenous meteorological covariates affect transformer forecasting performance for renewable energy?",
        "What are the practical computational costs of deploying transformer-based forecasters in operational grid management systems?",
    ],
    "keywords": [
        "transformer time series forecasting",
        "renewable energy generation prediction",
        "solar power forecasting attention models",
        "wind power deep learning forecasting",
        "long-horizon energy time series",
    ],
    "research_scope": "This research focuses on transformer-based deep learning architectures applied to univariate and multivariate time series forecasting of solar and wind renewable energy generation, spanning short-term (hourly) to long-term (multi-day) forecast horizons. The scope includes comparative evaluation against classical statistical methods (ARIMA, exponential smoothing) and recurrent neural network baselines (LSTM, GRU), as well as the incorporation of exogenous weather covariates such as irradiance, temperature, and wind speed. Excluded from scope are non-renewable energy load forecasting, hardware-level sensor design, and grid control/dispatch optimization algorithms, which are treated as downstream applications rather than core forecasting methodology.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of self-attention and transformer variants (Informer, Autoformer, FEDformer) for long-sequence renewable energy time series forecasting.",
        "Increasing use of hybrid architectures combining convolutional feature extraction with transformer temporal modeling for solar and wind forecasting.",
        "Rising interest in probabilistic and uncertainty-aware forecasting outputs to support grid operators with risk-aware decision making.",
        "Expanded use of multi-source data fusion, combining satellite imagery, numerical weather prediction outputs, and historical generation records.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of transformer forecasting robustness under abrupt weather regime shifts (e.g., sudden cloud cover or wind gust events) rather than average-case accuracy.",
            "based_on_papers": ["Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized cross-site benchmarking of transformer models across geographically diverse solar and wind farms with heterogeneous climate profiles.",
            "based_on_papers": ["Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: energy-aware and computational cost analysis of training and deploying large transformer forecasters relative to the carbon savings they enable through improved grid integration.",
            "based_on_papers": ["FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design lightweight or distilled transformer variants suited for edge deployment at distributed renewable energy sites.",
        "Develop uncertainty quantification methods that couple transformer point forecasts with calibrated prediction intervals for grid risk management.",
        "Explore transfer learning strategies that allow transformer forecasters trained on data-rich sites to generalize to newly commissioned solar and wind installations.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True}],
    "algorithms": [
        {
            "name": "Transformer (Informer)",
            "url": "https://arxiv.org/abs/2012.07436",
            "description": "A ProbSparse self-attention transformer architecture designed for efficient long-sequence time series forecasting, well suited to multi-day renewable generation horizons.",
            "verified": True,
        },
        {
            "name": "LSTM Baseline",
            "url": "https://arxiv.org/abs/1909.09586",
            "description": "A long short-term memory recurrent network used as a sequential baseline to benchmark transformer forecasting gains on solar and wind generation data.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)", "Continuous Ranked Probability Score (CRPS)"],
    "implementation_steps": [
        "Collect and preprocess historical solar/wind generation data alongside aligned meteorological covariates from public repositories",
        "Normalize and window the time series into fixed-length input-output sequences for supervised forecasting",
        "Implement baseline models (ARIMA, LSTM) and transformer variants (Informer, Autoformer) under a shared training pipeline",
        "Train each model across multiple forecast horizons using consistent train/validation/test splits stratified by season",
        "Evaluate models using MAE, RMSE, MAPE, and CRPS, with statistical significance testing across horizons",
        "Analyze attention weights and error patterns to interpret model behavior during weather regime transitions",
    ],
    "unverified_suggestions": ["NREL Solar Power Data for Integration Studies", "EIA Renewable Energy Generation Dataset", "WindToolkit"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates transformer architectures against well-established statistical and recurrent baselines.",
        "Research questions are specific and map directly to measurable evaluation metrics.",
        "The methodology includes a diverse set of forecast horizons, strengthening the generalizability of conclusions.",
        "Attention weight interpretability is incorporated as an analysis step, adding scientific value beyond raw accuracy comparison.",
    ],
    "weaknesses": [
        "The proposal does not fully justify why the reviewed papers' benchmark datasets are representative of diverse renewable energy site conditions, raising evidence sufficiency and dataset fit concerns.",
        "Computational cost and training resource requirements for transformer models are underspecified relative to baseline models.",
        "The proposal does not detail how missing or corrupted sensor data in real-world generation records will be handled.",
        "Cross-site generalization claims rely on a limited number of retrieved papers and would benefit from broader empirical validation.",
    ],
    "missing_components": [
        "An explicit ablation study plan isolating the contribution of exogenous weather covariates versus raw historical generation data.",
        "A discussion of deployment constraints (latency, memory footprint) for operational grid forecasting use cases.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based renewable energy forecasting, based on the retrieved abstract." for t in titles]
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
