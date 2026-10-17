import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Real-Time Sign Language Recognition and Translation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning architecture capable of recognizing continuous sign language gestures from video streams with low end-to-end latency.",
        "Design a translation pipeline that converts recognized sign sequences into grammatically correct spoken/written language sentences.",
        "Evaluate the trade-offs between recognition accuracy and inference speed for on-device, real-time deployment.",
        "Investigate the impact of multimodal input (hand pose, facial expression, body posture) on recognition robustness across diverse signers.",
    ],
    "research_questions": [
        "Which neural architectures (CNN-LSTM hybrids, transformers, graph neural networks on skeletal keypoints) achieve the best accuracy-latency trade-off for real-time sign language recognition?",
        "How does incorporating non-manual features such as facial expression and mouth shape improve translation quality compared to hand-gesture-only models?",
        "What data augmentation and transfer learning strategies best address the scarcity of large, labeled, signer-diverse sign language corpora?",
        "How well do current continuous sign language translation models generalize across different sign languages and signer demographics?",
    ],
    "keywords": [
        "sign language recognition",
        "continuous sign language translation",
        "gesture recognition deep learning",
        "skeletal pose estimation",
        "real-time video translation",
    ],
    "research_scope": "This research focuses on deep learning methods for recognizing and translating continuous sign language from RGB and depth video in real time, covering model architectures (CNN, RNN/LSTM, transformer, and graph-based skeletal models), multimodal fusion of manual and non-manual cues, and on-device inference constraints. It excludes purely rule-based or non-neural gesture recognition systems, static isolated fingerspelling recognition in the absence of continuous signing, and non-visual input modalities such as sEMG or glove-based sensors, focusing instead on camera-based, deployable solutions suitable for accessibility applications.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of skeletal keypoint representations (e.g., MediaPipe, OpenPose) combined with graph convolutional networks to reduce sensitivity to background and lighting variation.",
        "Increasing use of transformer-based sequence models for continuous sign language recognition and gloss-to-text translation, replacing earlier CNN-LSTM pipelines.",
        "Rising interest in multimodal fusion architectures that jointly model hand shape, facial expression, and body pose to capture non-manual grammatical markers.",
        "Emergence of lightweight, quantized, and knowledge-distilled models targeting real-time, on-device deployment on mobile and edge hardware.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: robust cross-signer generalization when training data is dominated by a small number of signers, leading to overfitting to individual signing styles.",
            "based_on_papers": ["Neural Sign Language Translation", "Skeleton-based Sign Language Recognition using Graph Convolutional Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarking of real-time inference latency and power consumption on consumer-grade mobile devices for continuous translation systems.",
            "based_on_papers": ["Real-time Sign Language Recognition using Temporal Convolutional Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of translation quality for grammatically complex sentences involving non-manual markers such as eyebrow raises and mouth morphemes.",
            "based_on_papers": ["Multi-modal Fusion for Continuous Sign Language Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Curating larger, signer-diverse, multi-dialect sign language video datasets with fine-grained gloss and non-manual annotations to improve generalization.",
        "Developing efficient transformer or state-space model variants specifically optimized for low-latency, on-device continuous sign language translation.",
        "Exploring self-supervised and few-shot learning approaches to reduce dependence on costly gloss-level annotation across multiple sign languages.",
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
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://arxiv.org/abs/1512.03385",
            "description": "Deep residual CNN backbone used to extract spatial features (hand shape, finger configuration, facial expression) from individual video frames before temporal modeling.",
            "verified": True,
        },
        {
            "name": "Long Short-Term Memory Network (LSTM)",
            "url": "https://arxiv.org/abs/1503.04069",
            "description": "Recurrent architecture applied over per-frame CNN or skeletal-pose features to model the temporal dynamics of continuous gesture sequences for sign recognition and gloss prediction.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Word Error Rate (WER)",
        "BLEU score",
        "Top-1 gesture classification accuracy",
        "End-to-end inference latency (ms per frame)",
    ],
    "implementation_steps": [
        "Collect and preprocess continuous sign language video data, extracting hand, body, and facial keypoints using a pose estimation library such as MediaPipe",
        "Normalize and augment keypoint and frame sequences to account for signer variability, lighting, and camera angle differences",
        "Train a spatial feature extractor (CNN) followed by a temporal sequence model (LSTM or transformer) on labeled gloss sequences",
        "Integrate a sequence-to-sequence translation module to convert predicted gloss sequences into grammatically fluent spoken-language text",
        "Optimize the trained model via quantization and pruning for real-time, low-latency inference on edge or mobile hardware",
        "Evaluate the end-to-end system on held-out signers using recognition accuracy, translation quality (BLEU), and inference latency benchmarks",
    ],
    "unverified_suggestions": ["RWTH-PHOENIX-Weather 2014T", "How2Sign", "MediaPipe Holistic"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly decomposes the problem into recognition and translation sub-tasks with well-defined objectives.",
        "The methodology combines spatial and temporal modeling in a manner consistent with established practice in continuous gesture recognition.",
        "Evaluation metrics appropriately span both linguistic translation quality (BLEU, WER) and system-level latency constraints relevant to real-time deployment.",
        "The identified research gaps around cross-signer generalization and non-manual marker modeling are well-grounded and actionable.",
    ],
    "weaknesses": [
        "The proposal does not specify how class imbalance across rare signs will be addressed during training.",
        "There is limited discussion of ethical and consent considerations for collecting video data of signers.",
        "The evidence base is drawn from a modest set of retrieved papers, so claims about state-of-the-art performance should be treated as provisional pending a broader literature sweep and dataset-fit verification.",
        "The plan does not detail a strategy for handling out-of-vocabulary signs during real-time translation.",
    ],
    "missing_components": [
        "A concrete data annotation protocol for gloss-level labeling and inter-annotator agreement measurement.",
        "A discussion of deployment hardware targets and their memory/compute budgets for real-time inference.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to real-time sign language recognition and translation, based on the retrieved abstract." for t in titles]
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
