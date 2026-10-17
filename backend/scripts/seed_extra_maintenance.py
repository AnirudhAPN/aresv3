import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Temporal Convolutional Networks for Predictive Maintenance in Manufacturing"

PLANNER_JSON = {
    "research_objectives": [
        "Develop temporal convolutional network (TCN) architectures that accurately predict remaining useful life (RUL) and impending failures of manufacturing equipment from multivariate sensor time series.",
        "Evaluate the trade-offs between prediction accuracy, receptive field size, and inference latency when deploying TCN-based predictive maintenance models on factory-floor edge hardware.",
        "Investigate dilated causal convolution designs and multi-scale feature extraction strategies to improve early fault detection in noisy, high-frequency industrial sensor streams.",
        "Assess the robustness of TCN-based predictive maintenance models to sensor drift, missing data, and operating-condition variability across different machine types.",
    ],
    "research_questions": [
        "How do TCN architectures compare to LSTM and GRU-based recurrent models in predicting remaining useful life from vibration, temperature, and current sensor data on industrial machinery?",
        "What dilation and kernel-size configurations best balance long-range temporal dependency capture against computational cost for real-time predictive maintenance on embedded controllers?",
        "How does the parallelizable, non-recurrent structure of TCNs affect training stability and inference throughput compared to recurrent architectures on large-scale industrial time series datasets?",
        "How well do TCN-based fault detectors generalize across different machine types, operating regimes, and sensor placements not represented in the training data?",
    ],
    "keywords": [
        "temporal convolutional networks",
        "predictive maintenance",
        "remaining useful life estimation",
        "industrial time series fault detection",
        "dilated causal convolutions",
    ],
    "research_scope": "This research focuses on temporal convolutional network methods for predictive maintenance in manufacturing settings, covering remaining useful life estimation and early fault/anomaly detection from multivariate sensor time series such as vibration, temperature, pressure, and motor current. The scope includes dilated causal convolution architectures, comparisons against recurrent and transformer-based sequence models, and evaluation on public bearing- and turbofan-degradation benchmarks as well as simulated factory sensor streams, with attention to deployment feasibility on resource-constrained programmable logic controllers and edge gateways. It excludes non-time-series predictive maintenance approaches such as static rule-based thresholding, maintenance scheduling optimization that does not rely on sensor-driven prediction, and domains outside discrete or process manufacturing (e.g., predictive maintenance for software systems).",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of dilated causal convolutions in TCNs to capture long-range temporal dependencies in sensor data without the sequential bottleneck of recurrent networks.",
        "Increasing use of public degradation benchmarks such as NASA C-MAPSS turbofan and bearing vibration datasets to standardize remaining-useful-life evaluation.",
        "Rising interest in hybrid architectures that combine TCN feature extractors with attention mechanisms to weight the most fault-relevant time steps.",
        "Emerging focus on lightweight TCN variants designed for deployment on programmable logic controllers and industrial edge gateways with limited compute.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: large-scale evaluation of TCN-based remaining-useful-life models when transferred across different machine types or sensor layouts without retraining.",
            "based_on_papers": ["Temporal Convolutional Networks for the Advance Prediction of ENSO"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic benchmarking of TCN inference latency and memory footprint on low-cost programmable logic controllers representative of real factory-floor deployments.",
            "based_on_papers": ["An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparative study of TCN robustness to missing or corrupted sensor readings caused by intermittent industrial network connectivity, rather than assuming clean, fully-sampled input streams.",
            "based_on_papers": ["Temporal Convolutional Networks for the Advance Prediction of ENSO", "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop transfer learning and domain adaptation methods that let TCN-based RUL models generalize across machine families and sensor configurations with minimal retraining.",
        "Design uncertainty-aware TCN outputs (e.g., prediction intervals) so maintenance planners can weigh confidence alongside point RUL estimates.",
        "Build standardized, multi-machine industrial benchmark datasets with realistic sensor dropout and noise to enable fairer cross-study comparison of predictive maintenance models.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository of ML datasets with standardized benchmarking metadata.", "verified": True}],
    "algorithms": [
        {
            "name": "Long Short-Term Memory (LSTM)",
            "url": "https://www.bioinf.jku.at/publications/older/2604.pdf",
            "description": "Recurrent architecture with gated memory cells, used as a baseline sequence model for comparing against TCNs on remaining-useful-life and fault prediction from sensor time series.",
            "verified": True,
        },
        {
            "name": "Temporal Convolutional Network (TCN)",
            "url": "https://arxiv.org/abs/1803.01271",
            "description": "Dilated causal convolutional architecture that captures long-range temporal dependencies in parallel rather than sequentially, forming the core model family for predictive maintenance from industrial sensor streams.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "Precision", "Recall"],
    "implementation_steps": [
        "Collect and preprocess multivariate sensor time series (vibration, temperature, current) from target machinery, labeling run-to-failure sequences with remaining useful life or fault onset timestamps.",
        "Normalize and window the sensor streams into fixed-length input sequences, handling missing samples and aligning sampling rates across heterogeneous sensor channels.",
        "Split data into training, validation, and test sets by machine unit or run-to-failure trajectory to avoid temporal leakage between training and evaluation.",
        "Train TCN architectures with varying dilation rates and receptive field sizes, tuning against LSTM and GRU baselines on the same windowed sequences.",
        "Evaluate trained models on held-out run-to-failure trajectories using RMSE, MAE, precision, and recall for RUL regression and fault-onset classification.",
        "Benchmark model size, inference latency, and memory footprint on representative edge or programmable logic controller hardware to assess factory-floor deployability.",
    ],
    "unverified_suggestions": ["NASA C-MAPSS Turbofan Degradation Dataset", "Case Western Reserve University Bearing Dataset", "PHM 2012 Bearing Prognostics Dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates TCNs over recurrent baselines by tying the parallelizable dilated-convolution structure to real industrial constraints such as inference latency on edge hardware.",
        "The methodology specifies machine-unit-level train/test splitting, which correctly avoids temporal leakage common in run-to-failure prognostics datasets.",
        "Evaluation metrics combine regression (RMSE, MAE for RUL) and classification (precision, recall for fault onset), appropriate for the dual nature of predictive maintenance tasks.",
        "The scope explicitly excludes non-sensor-driven maintenance approaches, keeping the research well-focused on the TCN modeling contribution.",
    ],
    "weaknesses": [
        "The evidence sufficiency for claimed gaps is limited, as it is based on a small retrieved sample of 8 papers rather than a comprehensive systematic review of the prognostics literature.",
        "The proposal does not fully address dataset fit, since suggested datasets like C-MAPSS are simulated turbofan data and may not transfer cleanly to discrete manufacturing equipment such as CNC machines or motors.",
        "Benchmarking on programmable logic controller hardware is proposed but lacks specificity on which controller models or compute budgets will be tested.",
        "The gap around missing or corrupted sensor readings is identified but no concrete imputation or robustness-training approach is specified in this plan.",
    ],
    "missing_components": [
        "A plan for validating predicted remaining-useful-life estimates against maintenance technician ground truth or actual failure logs on newly collected field data.",
        "An ablation study isolating the contribution of dilation rate and receptive field size versus overall network depth to final RUL prediction accuracy.",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.77,
}

CONFIDENCE_JSON = {"internal_consistency": 0.85, "specificity": 0.82, "completeness": 0.87}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to temporal convolutional networks for predictive maintenance in manufacturing, based on the retrieved abstract." for t in titles]
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
