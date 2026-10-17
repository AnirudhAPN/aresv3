import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Networks for Real-Time Hand Gesture Recognition in Human-Robot Interaction"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a real-time neural network pipeline for recognizing dynamic hand gestures from RGB or depth camera streams for human-robot interaction commands",
        "Evaluate lightweight architectures, including temporal convolutional networks and compact keypoint-based classifiers, balancing recognition accuracy against inference latency for real-time robotic control",
        "Investigate skeleton/keypoint-based versus raw-image-based gesture recognition approaches for robustness and computational efficiency",
        "Assess model robustness to variation in gesture speed, partial occlusion, and background clutter typical of unconstrained human-robot interaction environments",
    ],
    "research_questions": [
        "Can lightweight temporal neural architectures achieve real-time gesture recognition (targeting 15-30 FPS) suitable for closed-loop robot control?",
        "How does skeleton/keypoint-based gesture recognition compare to raw RGB frame-based CNN approaches in accuracy and latency?",
        "What is the effect of partial hand occlusion, common during natural physical human-robot interaction, on recognition accuracy across architectures?",
        "How well do gesture recognition models trained on curated datasets generalize to unscripted, spontaneous gestures produced during live human-robot interaction sessions?",
    ],
    "keywords": [
        "hand gesture recognition deep learning",
        "real-time gesture recognition human-robot interaction",
        "skeleton-based gesture recognition",
        "temporal convolutional network gesture",
        "hand keypoint detection robotics",
    ],
    "research_scope": (
        "This research focuses on vision-based dynamic hand gesture recognition for human-robot interaction command "
        "input, covering both skeleton/keypoint-based and raw-frame CNN-based recognition pipelines, temporal "
        "sequence modeling architectures (temporal convolutional networks, lightweight transformers), and latency "
        "benchmarking suitable for real-time closed-loop robot control. It includes robustness evaluation under "
        "partial occlusion, gesture speed variation, and background clutter. The scope excludes full-body pose-based "
        "activity recognition, sign language recognition (a related but linguistically distinct task), and haptic or "
        "wearable-sensor-based gesture input, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Shift toward skeleton/keypoint-based gesture recognition pipelines built on real-time hand-tracking models for computational efficiency over raw-frame CNNs",
        "Growing use of temporal convolutional networks and lightweight transformers for real-time dynamic gesture sequence classification",
        "Adoption of multimodal RGB-D sensing to improve robustness to lighting and background clutter in gesture recognition for human-robot interaction",
        "Increasing focus on edge deployment on embedded robot-class compute, requiring model compression and quantization for gesture recognition pipelines",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: benchmarking of gesture recognition latency and accuracy tradeoffs specifically on embedded robot-class hardware (e.g., Jetson-class GPUs) rather than only desktop GPU evaluation.",
            "based_on_papers": ["MediaPipe Hands: On-device Real-time Hand Tracking"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of recognition robustness under partial hand occlusion scenarios that arise naturally during physical human-robot interaction, such as when a robot gripper or the user's own body partially blocks the camera view.",
            "based_on_papers": ["Temporal Convolutional Networks for Action Segmentation and Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantified generalization gap between models trained on curated gesture datasets and spontaneous, unscripted gestures produced during live, unconstrained interaction sessions.",
            "based_on_papers": ["MediaPipe Hands: On-device Real-time Hand Tracking"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish standardized embedded-hardware latency/accuracy benchmarks for gesture recognition models targeting real-time robot control",
        "Develop occlusion-robust training protocols using synthetic or physically staged partial-occlusion augmentation",
        "Collect and benchmark on spontaneous, unscripted gesture datasets captured during live human-robot interaction sessions rather than only scripted gesture recordings",
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
            "name": "Temporal Convolutional Network (TCN)",
            "url": "https://arxiv.org/abs/1803.01271",
            "description": "Causal, dilated convolutional architecture applied to sequences of extracted hand keypoint features to classify dynamic gestures with low inference latency suitable for real-time control.",
            "verified": True,
        },
        {
            "name": "MediaPipe Hands",
            "url": "https://arxiv.org/abs/2006.10214",
            "description": "Real-time, on-device hand landmark detection model used as the keypoint extraction front-end, converting raw RGB frames into per-frame hand skeleton representations for downstream gesture classification.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Recognition accuracy",
        "F1 score per gesture class",
        "Inference latency (ms per frame)",
        "Frames-per-second (FPS) throughput",
    ],
    "implementation_steps": [
        "Collect and label a dynamic hand gesture video dataset relevant to human-robot interaction commands (e.g., stop, come, point, wave)",
        "Extract per-frame hand keypoints using a real-time hand-tracking model to produce skeleton feature sequences",
        "Construct fixed-length temporal windows of keypoint features per gesture instance",
        "Train a TCN or lightweight transformer classifier on the keypoint sequence representations",
        "Train a raw-RGB CNN baseline for direct comparison against the keypoint-based pipeline",
        "Benchmark accuracy and latency on embedded robot-class hardware and evaluate robustness under simulated partial-occlusion perturbation tests",
    ],
    "unverified_suggestions": ["Jester Hand Gesture Dataset", "NVIDIA Dynamic Hand Gesture Dataset", "IPN Hand Dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets real-time deployment constraints (FPS, latency) rather than only offline classification accuracy, aligning the evaluation with the actual HRI use case",
        "Comparing skeleton-based and raw-RGB pipelines gives a clear, well-motivated ablation of the accuracy-efficiency tradeoff central to the research questions",
        "Including embedded-hardware benchmarking directly addresses a practical deployment gap often overlooked in gesture recognition research evaluated only on desktop GPUs",
        "The occlusion robustness evaluation is grounded in a realistic HRI scenario (robot gripper or user body blocking the camera) rather than generic corruption benchmarks",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how simulated partial-occlusion perturbations will be validated as representative of real occlusion patterns encountered during physical interaction",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host video-based gesture datasets, so feasibility depends entirely on the unverified, domain-specific dataset suggestions",
        "The plan does not address how gesture recognition will be integrated with a rejection/uncertainty mechanism to avoid triggering unintended robot actions on ambiguous or out-of-vocabulary hand motions",
    ],
    "missing_components": [
        "An explicit out-of-vocabulary / ambiguous-gesture rejection mechanism to prevent unintended robot actions",
        "A user study protocol for evaluating recognition performance with real users performing spontaneous, unscripted gestures rather than only pre-recorded datasets",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.72,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.87}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to real-time hand gesture recognition for human-robot interaction, based on the retrieved abstract." for t in titles]
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
