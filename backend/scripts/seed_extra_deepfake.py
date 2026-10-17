import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Networks for Real-Time Deepfake Video Detection"

PLANNER_JSON = {
    "research_objectives": [
        "Design a neural network architecture capable of detecting deepfake manipulations in video streams within real-time latency constraints.",
        "Identify and exploit spatial and temporal artifacts characteristic of face-swap and reenactment generative models to improve detection accuracy.",
        "Evaluate the trade-off between detection accuracy and inference speed across candidate architectures for deployment on consumer and edge hardware.",
        "Assess the generalization of trained detectors to unseen deepfake generation methods not present in the training distribution.",
    ],
    "research_questions": [
        "Can a lightweight convolutional or transformer-based architecture achieve real-time (>=24 fps) inference while maintaining competitive frame-level detection accuracy?",
        "Which spatiotemporal artifacts (e.g., blending boundaries, eye-blink irregularities, frame-to-frame flicker) are most discriminative for distinguishing deepfake video from authentic video?",
        "How well do detectors trained on one family of generative models (e.g., autoencoder-based face swaps) generalize to video manipulated with a different family (e.g., diffusion-based reenactment)?",
        "What is the impact of video compression artifacts on the reliability of real-time deepfake detectors deployed on compressed social-media video streams?",
    ],
    "keywords": [
        "deepfake video detection",
        "real-time facial forgery detection",
        "spatiotemporal artifact detection",
        "generalizable forgery detection",
        "lightweight CNN video classification",
    ],
    "research_scope": (
        "This research focuses on neural network-based detection of face manipulation (deepfake) content in video, "
        "covering frame-level and clip-level architectures that combine spatial artifact detection with temporal "
        "consistency modeling, with an explicit emphasis on inference-time constraints suitable for real-time or "
        "near-real-time deployment (e.g., live-stream moderation, video-call authentication). It covers detection "
        "of face-swap, expression-reenactment, and lip-sync manipulation techniques, and considers robustness to "
        "common video compression codecs. It excludes audio-only deepfake (voice cloning) detection, excludes "
        "detection of fully AI-generated (text-to-video) synthetic video without an underlying real identity, and "
        "excludes watermarking or provenance-based (non-neural) authentication approaches, which are treated as "
        "out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of two-stream architectures that fuse a spatial artifact-detection branch (operating on individual frames) with a temporal-consistency branch (operating on optical flow or frame sequences).",
        "Increasing adoption of vision transformer backbones for forgery detection, replacing pure CNN encoders to better capture long-range spatial dependencies in manipulated regions.",
        "Rising interest in frequency-domain analysis (e.g., DCT or Fourier spectra) to expose GAN-specific upsampling artifacts that are less visible in the pixel domain.",
        "Shift toward cross-dataset and cross-manipulation generalization benchmarks, reflecting concern that detectors overfit to artifacts of the specific generator used during training.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic benchmarking of end-to-end inference latency and throughput on commodity edge hardware (e.g., mobile NPUs) rather than server-class GPUs.",
            "based_on_papers": ["FaceForensics++: Learning to Detect Manipulated Facial Images", "Xception: Deep Learning with Depthwise Separable Convolutions"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness evaluation against adaptive adversaries who deliberately optimize the deepfake generator to evade a known detector (adversarial deepfake generation).",
            "based_on_papers": ["Face X-Ray for More General Face Forgery Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantifying detection accuracy degradation as a function of video compression bitrate on real social-media re-encoding pipelines rather than lab-controlled compression levels.",
            "based_on_papers": ["FaceForensics++: Learning to Detect Manipulated Facial Images", "Face X-Ray for More General Face Forgery Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Explore knowledge distillation from large forgery-detection transformers into compact student networks optimized for real-time edge inference.",
        "Investigate self-supervised pretraining on unlabeled real video to learn generic temporal-consistency representations that transfer across unseen manipulation types.",
        "Study continual-learning strategies that allow deployed detectors to incorporate artifacts from newly emerging deepfake generation methods without full retraining.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "COCO (Common Objects in Context)",
            "url": "https://cocodataset.org/",
            "description": "Object detection, segmentation, and captioning benchmark dataset.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://arxiv.org/abs/1610.02357",
            "description": "A depthwise-separable convolutional architecture (e.g., XceptionNet) commonly used as a frame-level spatial artifact classifier in deepfake detection pipelines, offering a favorable accuracy-to-latency trade-off for real-time use.",
            "verified": True,
        },
        {
            "name": "Recurrent Convolutional Network (CNN-LSTM / CNN-GRU)",
            "url": "https://arxiv.org/abs/1806.08893",
            "description": "A hybrid architecture that feeds per-frame CNN features into a recurrent temporal module to detect frame-to-frame inconsistencies such as flicker and unnatural motion, characteristic of face-swap generation artifacts.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Frame-level accuracy",
        "Area under the ROC curve (AUC)",
        "Equal error rate (EER)",
        "Inference latency (frames per second) on target hardware",
    ],
    "implementation_steps": [
        "Collect and preprocess labeled real and deepfake video clips, extracting and aligning cropped face regions per frame",
        "Apply data augmentation reflecting realistic deployment conditions, including compression, resizing, and color jitter",
        "Train a spatial artifact-detection backbone on individual frames to classify real versus manipulated faces",
        "Extend the backbone with a lightweight temporal module (recurrent or 3D-convolutional) to capture inter-frame inconsistencies across short clips",
        "Optimize the trained model for inference through quantization or pruning and benchmark latency and throughput on target real-time hardware",
        "Evaluate cross-manipulation and cross-dataset generalization by testing on deepfake generation methods excluded from the training set",
    ],
    "unverified_suggestions": ["FaceForensics++", "Celeb-DF", "DFDC (Deepfake Detection Challenge dataset)"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly frames the real-time latency constraint as a first-class design requirement rather than an afterthought, shaping architecture choices from the outset.",
        "The two-branch (spatial plus temporal) methodology is well-grounded in cited architectures and directly targets known deepfake artifact types.",
        "The research questions are specific and testable, particularly around cross-manipulation generalization and compression robustness.",
        "The scope explicitly excludes audio-only and provenance-based approaches, keeping the research focus tractable.",
    ],
    "weaknesses": [
        "The proposal does not specify how much labeled data would be needed to reach reliable generalization estimates, making it hard to assess evidence sufficiency for the cross-manipulation claims.",
        "Dataset fit is uncertain: the whitelisted COCO dataset is not a natural source for face-forgery video, so the proposed unverified suggestions (FaceForensics++, Celeb-DF, DFDC) would need independent verification before use.",
        "No ablation plan is specified for isolating the individual contribution of the spatial branch versus the temporal branch to overall detection accuracy.",
        "The proposal lacks discussion of statistical significance testing when comparing AUC or EER across candidate architectures.",
    ],
    "missing_components": [
        "A discussion of the compute and power budget assumed for the target real-time edge deployment scenario.",
        "An explicit adversarial robustness evaluation protocol against generators optimized to evade the trained detector.",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to real-time neural network-based deepfake video detection, based on the retrieved abstract." for t in titles]
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
