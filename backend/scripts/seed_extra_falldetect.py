import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Lightweight Neural Networks for Real-Time Fall Detection in Elderly Care"

PLANNER_JSON = {
    "research_objectives": [
        "Design a lightweight neural network architecture capable of detecting falls in real time on resource-constrained wearable and edge devices.",
        "Minimize inference latency and energy consumption while preserving high fall-detection sensitivity and specificity.",
        "Evaluate the trade-off between model compression techniques (pruning, quantization, distillation) and detection accuracy in elderly care settings.",
        "Assess the robustness of the proposed system against false alarms triggered by activities of daily living that resemble falls.",
    ],
    "research_questions": [
        "What lightweight neural network architectures achieve the best accuracy-to-latency ratio for real-time fall detection on embedded or wearable hardware?",
        "How do model compression techniques such as quantization and pruning affect fall-detection sensitivity in elderly populations?",
        "Can multimodal sensor fusion (accelerometer, gyroscope, and ambient sensors) improve detection reliability without significantly increasing model size?",
        "What is the impact of on-device inference versus cloud-based inference on detection latency and user privacy?",
    ],
    "keywords": [
        "fall detection",
        "lightweight neural networks",
        "wearable sensors",
        "edge inference",
        "elderly care monitoring",
    ],
    "research_scope": "This research focuses on the design and evaluation of lightweight neural network models for real-time fall detection targeted at elderly individuals in home and assisted-living environments. The scope encompasses model architecture selection (e.g., compact CNNs, depthwise-separable convolutions, and recurrent variants), compression strategies for deployment on wearable and edge devices with limited compute and battery capacity, and evaluation using inertial sensor datasets. The study will examine detection accuracy, inference latency, energy efficiency, and false-alarm rates, while excluding video-based or camera-based fall detection approaches due to privacy concerns in domestic settings.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of depthwise-separable and MobileNet-style convolutions to shrink model footprints for wearable fall-detection systems.",
        "Growing adoption of knowledge distillation to transfer accuracy from large threshold-based or deep LSTM models into compact student networks.",
        "Rising interest in sensor fusion combining accelerometer and gyroscope streams to reduce false positives from abrupt but non-fall movements.",
        "Shift toward on-device TinyML deployment using frameworks such as TensorFlow Lite Micro to avoid cloud latency and preserve privacy.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of quantization-aware training specifically for fall-detection models under real elderly-care noise conditions rather than benchmark datasets collected from young adults.",
            "based_on_papers": ["A Deep Learning Approach for Fall Detection Using Wearable Sensors", "Lightweight CNN for Human Activity Recognition on Edge Devices"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: long-term robustness testing of lightweight fall-detection models against sensor drift and battery-induced sampling-rate variation over multi-month deployments.",
            "based_on_papers": ["Real-Time Fall Detection System Using Accelerometer and Machine Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparative analysis of federated learning approaches that would allow lightweight fall-detection models to personalize to an individual's gait and movement patterns without transmitting raw sensor data off-device.",
            "based_on_papers": ["Edge Computing for Healthcare Monitoring: A Survey"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Investigate personalized, federated on-device fine-tuning to adapt lightweight fall-detection models to individual gait patterns while preserving privacy.",
        "Develop standardized elderly-specific benchmark datasets that capture realistic fall and near-fall events alongside common daily activities to reduce false-alarm rates.",
        "Explore neural architecture search constrained by microcontroller memory and energy budgets to automatically discover optimal lightweight fall-detection topologies.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "PhysioNet", "url": "https://physionet.org/", "description": "Curated clinical and physiological signal datasets for healthcare research.", "verified": True}
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://www.tensorflow.org/tutorials/images/cnn",
            "description": "A deep learning architecture that applies convolutional filters to extract spatial and temporal patterns from accelerometer and gyroscope time-series windows, well suited for classifying fall versus non-fall movement signatures.",
            "verified": True,
        },
        {
            "name": "MobileNetV2 (Depthwise-Separable Convolutions)",
            "url": "https://arxiv.org/abs/1801.04381",
            "description": "A lightweight convolutional architecture that factorizes standard convolutions into depthwise and pointwise operations, drastically reducing parameter count and multiply-accumulate operations for efficient on-device inference.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Sensitivity (Recall) for true fall events",
        "Specificity (True Negative Rate) for non-fall activities",
        "F1-score balancing precision and recall",
        "Inference latency (milliseconds per prediction) on target embedded hardware",
    ],
    "implementation_steps": [
        "Collect and preprocess accelerometer and gyroscope time-series data, segmenting it into fixed-length sliding windows around candidate fall events",
        "Apply data augmentation and normalization to account for sensor placement variability and elderly-specific movement speeds",
        "Design and train a baseline lightweight CNN architecture using depthwise-separable convolutions for feature extraction",
        "Apply post-training quantization and structured pruning to compress the trained model for deployment on microcontroller-class hardware",
        "Deploy the compressed model to a wearable prototype and benchmark real-time inference latency, memory footprint, and energy consumption",
        "Evaluate detection performance against held-out elderly activity data, tuning decision thresholds to minimize false alarms while preserving sensitivity",
    ],
    "unverified_suggestions": [
        "Consider incorporating a lightweight LSTM or GRU layer to capture temporal dependencies across longer movement sequences, though this may increase latency and should be validated empirically.",
        "Explore synthetic fall data generation via simulation or generative models to supplement limited real-world elderly fall datasets, pending validation of realism.",
        "Investigate edge-cloud hybrid inference where ambiguous cases are escalated to a cloud model, though this introduces latency and connectivity dependencies not yet benchmarked.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The research plan clearly targets a well-motivated real-world healthcare problem with concrete deployment constraints (latency, memory, energy).",
        "The methodology combines architecture design with practical compression techniques, reflecting realistic embedded deployment considerations.",
        "Evaluation metrics appropriately balance detection sensitivity against false-alarm minimization, which is critical for elderly-care usability.",
        "The identified gaps around personalization and long-term robustness point to meaningful, underexplored research directions.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small, limiting confidence that all major competing lightweight architectures have been considered.",
        "The proposal does not fully address how the chosen datasets generalize to elderly-specific gait and fall dynamics versus younger benchmark populations.",
        "There is limited discussion of privacy-preserving data collection protocols for continuous in-home monitoring.",
        "The plan does not specify a clear baseline comparison against non-neural threshold-based fall-detection methods still common in commercial devices.",
    ],
    "missing_components": [
        "A defined plan for user studies or clinical validation involving actual elderly participants to assess real-world false-alarm tolerance.",
        "An explicit power/latency budget target tied to specific commercial wearable hardware specifications.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to lightweight real-time fall detection for elderly care, based on the retrieved abstract." for t in titles]
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
