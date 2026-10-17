import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Networks for Precipitation Nowcasting Using Radar Imagery"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning model that predicts short-term (0-6 hour) precipitation intensity fields directly from sequences of radar reflectivity images",
        "Compare convolutional and recurrent spatiotemporal architectures for their ability to capture the motion, growth, and decay of precipitation cells",
        "Evaluate whether learned nowcasting models outperform traditional optical-flow-based extrapolation methods on lead times beyond 1 hour",
        "Assess the sensitivity of nowcasting accuracy to radar data resolution, temporal sampling interval, and geographic region",
    ],
    "research_questions": [
        "Can convolutional and recurrent neural network architectures trained on historical radar reflectivity sequences produce more accurate short-term precipitation forecasts than optical-flow extrapolation baselines?",
        "How does forecast skill degrade as lead time increases from 10 minutes to 6 hours, and which architectural components most affect this degradation?",
        "What is the impact of incorporating multi-channel radar products (reflectivity, Doppler velocity) versus single-channel reflectivity-only inputs on nowcasting accuracy?",
        "How well do models trained on one geographic region and radar network generalize to precipitation patterns in a different region?",
    ],
    "keywords": [
        "precipitation nowcasting",
        "radar reflectivity imagery",
        "convolutional LSTM",
        "spatiotemporal forecasting",
        "extreme weather prediction",
    ],
    "research_scope": "This research focuses on data-driven precipitation nowcasting using sequences of ground-based weather radar reflectivity images as input, targeting lead times from a few minutes up to six hours. The scope covers the design and comparison of convolutional neural network and convolutional-recurrent (e.g., ConvLSTM) architectures trained end-to-end to predict future radar reflectivity or derived precipitation intensity fields, evaluated against classical optical-flow extrapolation baselines using standard verification metrics such as critical success index and probability of detection. The study is limited to radar-only inputs from open meteorological archives and does not incorporate numerical weather prediction model output, satellite imagery fusion, or citizen-science rain gauge networks, though these are identified as promising extensions.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of ConvLSTM and its variants (TrajGRU, PredRNN) for radar-based nowcasting, consistently outperforming pixel-wise optical-flow extrapolation at short lead times",
        "Growing interest in generative approaches (e.g., GAN-based and diffusion-based nowcasting models such as DGMR) to counter the blurring of predicted precipitation fields at longer lead times",
        "Increasing use of large-scale open radar datasets (e.g., MRMS, HKO-7, SEVIR) to benchmark and pretrain nowcasting models",
        "Shift toward hybrid architectures that combine physically motivated advection terms with learned residual corrections",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of how prediction blurring and loss of small-scale convective detail at longer lead times differs across loss functions (MSE, adversarial, perceptual) under a controlled, identical backbone architecture.",
            "based_on_papers": ["Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: limited evaluation of cross-region generalization, i.e., whether a nowcasting model trained on radar data from one climate zone or radar network retains skill when applied to a different geographic region without retraining.",
            "based_on_papers": ["Skilful Precipitation Nowcasting Using Deep Generative Models of Radar"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: insufficient attention to calibrated uncertainty quantification for extreme, high-intensity precipitation events, where reliable probabilistic forecasts matter most for flash-flood warning systems.",
            "based_on_papers": ["Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop loss functions or training curricula that explicitly preserve small-scale convective structure at longer lead times rather than regressing to the conditional mean",
        "Build standardized cross-region benchmarks to evaluate the transferability of radar-based nowcasting models across different climates and radar hardware",
        "Investigate probabilistic and ensemble nowcasting architectures that produce calibrated uncertainty estimates specifically for high-intensity precipitation extremes",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "Copernicus Open Access Hub",
            "url": "https://scihub.copernicus.eu/",
            "description": "Satellite earth-observation data, commonly used for climate research.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "description": "Used as a per-frame spatial feature extractor and as the backbone for encoder-decoder nowcasting architectures that map input radar frames to predicted future reflectivity fields.",
            "verified": True,
        },
        {
            "name": "Convolutional LSTM (ConvLSTM)",
            "url": "https://arxiv.org/abs/1506.04214",
            "description": "Spatiotemporal recurrent architecture that replaces the fully-connected gates of a standard LSTM with convolutional operations, enabling it to jointly model the spatial structure and temporal evolution of radar echo sequences for nowcasting.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Critical Success Index (CSI)",
        "Probability of Detection (POD)",
        "False Alarm Ratio (FAR)",
        "Root Mean Squared Error (RMSE)",
    ],
    "implementation_steps": [
        "Collect and preprocess historical radar reflectivity sequences from an open archive (e.g., NEXRAD MRMS composites), regridding to a common spatial resolution and normalizing intensity values",
        "Construct input-output sequence pairs by sampling sliding windows of past radar frames (e.g., 10-20 minute intervals) paired with future frames at multiple lead times",
        "Implement and train a ConvLSTM-based encoder-decoder model to predict future radar reflectivity frames from the input sequence, using a combination of MSE and CSI-weighted loss to emphasize heavier precipitation",
        "Implement an optical-flow extrapolation baseline (e.g., Lucas-Kanade or Dense Rotational-flow-based advection) for comparison",
        "Train and validate models on a held-out temporal split, tuning hyperparameters such as sequence length, hidden state dimensionality, and learning rate on a validation set",
        "Evaluate all models across multiple lead times (10 minutes to 6 hours) and precipitation intensity thresholds using CSI, POD, FAR, and RMSE, and assess generalization on radar data from a held-out geographic region",
    ],
    "unverified_suggestions": [
        "Consider fusing satellite infrared imagery with radar reflectivity as an auxiliary input channel to extend forecast skill in regions with sparse radar coverage",
        "Explore adversarial or diffusion-based generative losses to reduce blurring of predicted precipitation fields at longer lead times",
        "Investigate transfer learning or domain adaptation techniques to improve cross-region generalization without full retraining on the target radar network",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates the use of spatiotemporal deep learning to overcome known limitations of optical-flow extrapolation at longer lead times",
        "The evaluation plan uses meteorologically standard verification metrics (CSI, POD, FAR) alongside RMSE, enabling direct comparison with the operational nowcasting literature",
        "Including a classical optical-flow baseline provides a meaningful reference point for judging the added value of the learned model",
        "The research questions explicitly probe forecast skill degradation over lead time, which is the central practical challenge in nowcasting",
    ],
    "weaknesses": [
        "The evidence base is limited to 8 retrieved papers, which may not sufficiently cover recent generative and diffusion-based nowcasting approaches or the largest published benchmark datasets",
        "The methodology does not specify how class imbalance between light and extreme precipitation events will be handled during training, despite extreme events being operationally most critical",
        "No explicit plan is given for statistical significance testing when comparing the proposed model against the optical-flow baseline across lead times",
        "Reliance on a single regional radar archive for training and validation limits confidence that reported skill will transfer to other climates or radar hardware, an issue the proposal itself later flags as a gap",
    ],
    "missing_components": [
        "An ablation study isolating the contribution of the recurrent (ConvLSTM) component versus a purely convolutional encoder-decoder baseline",
        "A discussion of inference latency and computational cost relative to the real-time operational requirements of a nowcasting system",
    ],
    "novelty_score": 0.66,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to neural network-based precipitation nowcasting from radar imagery, based on the retrieved abstract." for t in titles]
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
