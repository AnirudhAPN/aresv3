import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Real-Time Weapon Detection in Public Surveillance Video"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning-based object detection system capable of identifying firearms and edged weapons in real-time surveillance video streams.",
        "Evaluate the trade-off between detection accuracy and inference latency for deployment on edge and embedded surveillance hardware.",
        "Investigate methods for reducing false positive rates in cluttered, low-resolution, and low-light public surveillance footage.",
        "Assess the generalizability of trained models across varied camera angles, occlusion levels, and weapon types.",
    ],
    "research_questions": [
        "Which deep learning architectures achieve the best balance of detection accuracy and real-time inference speed for weapon detection in CCTV footage?",
        "How does model performance degrade under real-world surveillance conditions such as poor lighting, motion blur, and partial occlusion?",
        "What data augmentation and training strategies most effectively reduce false positive alerts in public surveillance deployments?",
        "How can weapon detection models be optimized for deployment on resource-constrained edge devices without significant accuracy loss?",
    ],
    "keywords": [
        "real-time weapon detection",
        "surveillance video object detection",
        "firearm detection deep learning",
        "edge deployment CNN",
        "CCTV anomaly detection",
    ],
    "research_scope": "This research focuses on the design, training, and evaluation of deep learning-based object detection systems for identifying firearms and other weapons in real-time public surveillance video streams. The scope encompasses convolutional neural network and single-shot detector architectures suited to low-latency inference, strategies for mitigating false positives caused by challenging real-world conditions (occlusion, poor lighting, camera angle variation), and the practical constraints of deploying such models on edge computing hardware commonly used in CCTV infrastructure. The work does not address post-detection response systems, legal or privacy policy frameworks, or non-visual sensing modalities such as millimeter-wave scanning.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of single-stage detectors such as YOLO variants for real-time weapon detection due to their favorable speed-accuracy trade-off.",
        "Growing use of transfer learning from large-scale object detection benchmarks (e.g., COCO) followed by fine-tuning on smaller, weapon-specific datasets.",
        "Increasing interest in lightweight and quantized network architectures to enable deployment on edge devices co-located with surveillance cameras.",
        "Use of synthetic data generation and augmentation to compensate for the scarcity of labeled weapon imagery in realistic surveillance conditions.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of weapon detection models under long-term real-world CCTV deployment conditions, including sensor degradation, seasonal lighting changes, and compression artifacts from video streaming pipelines.",
            "based_on_papers": ["Real-Time Gun Detection in CCTV: An Open Problem", "Automatic Handgun Detection Alarm in Videos Using Deep Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized, large-scale, publicly available benchmark datasets that combine diverse weapon types, camera viewpoints, and occlusion scenarios comparable in scale to general object detection benchmarks like COCO.",
            "based_on_papers": ["Weapon Detection Using Deep Learning: A Survey"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of false positive rates and their operational impact (e.g., alert fatigue among human security operators) when models are deployed at scale across multi-camera surveillance networks.",
            "based_on_papers": ["Automatic Handgun Detection Alarm in Videos Using Deep Learning", "Weapon Detection Using Deep Learning: A Survey"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Development of temporally-aware detection models that leverage multi-frame video context rather than single-frame inference to improve robustness and reduce false positives.",
        "Creation of standardized, ethically sourced, and diverse weapon detection benchmark datasets to enable fair cross-study comparison.",
        "Exploration of federated or on-device learning approaches that allow surveillance systems to adapt to site-specific conditions without centralizing sensitive video data.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "COCO (Common Objects in Context)", "url": "https://cocodataset.org/", "description": "Object detection, segmentation, and captioning benchmark dataset.", "verified": True}],
    "algorithms": [
        {
            "name": "YOLO (You Only Look Once)",
            "url": "https://arxiv.org/abs/1506.02640",
            "description": "A single-stage object detection architecture that frames detection as a regression problem, enabling real-time inference speeds suitable for live surveillance video streams.",
            "verified": True,
        },
        {
            "name": "ResNet (Residual Network) Backbone",
            "url": "https://arxiv.org/abs/1512.03385",
            "description": "A deep convolutional neural network backbone using residual connections to enable stable training of very deep feature extractors, commonly used as the backbone for detection heads.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "mean Average Precision (mAP) at IoU thresholds 0.5 and 0.5:0.95",
        "Frames Per Second (FPS) for real-time inference throughput",
        "False Positive Rate under varied lighting and occlusion conditions",
        "Precision-Recall curve and F1-score for weapon class detection",
    ],
    "implementation_steps": [
        "Collect and curate a labeled dataset of firearm and edged-weapon images and video frames from public surveillance-like sources, supplemented with COCO-pretrained background classes.",
        "Apply data augmentation techniques (random occlusion, brightness/contrast jitter, motion blur simulation) to approximate real-world CCTV conditions.",
        "Fine-tune a pretrained single-stage detector (e.g., YOLO with a ResNet or CSPDarknet backbone) on the curated weapon dataset using transfer learning.",
        "Optimize the trained model for edge deployment via quantization and pruning, and benchmark inference latency on representative embedded hardware.",
        "Evaluate detection performance on a held-out test set spanning varied lighting, occlusion, and camera angle conditions, reporting mAP, FPS, and false positive rate.",
        "Conduct an ablation study comparing backbone architectures and input resolutions to characterize the accuracy-latency trade-off curve.",
    ],
    "unverified_suggestions": [
        "Incorporating multi-frame temporal fusion (e.g., 3D convolutions or recurrent layers) to reduce single-frame false positives, though this has not been benchmarked against the single-frame baselines cited here.",
        "Using synthetic weapon imagery generated via game engines or diffusion models to augment scarce real-world training data, an approach not directly validated in the retrieved literature.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly targets a real-time inference constraint, aligning the choice of single-stage detectors with the practical deployment context of surveillance systems.",
        "The methodology grounds algorithm selection in well-established, verifiable architectures (YOLO, ResNet) with clear provenance.",
        "The evaluation plan includes both accuracy metrics (mAP, F1) and operational metrics (FPS, false positive rate), reflecting a realistic deployment-oriented perspective.",
        "The identified research gaps around long-term deployment robustness and benchmark standardization are well-motivated and grounded in the reviewed literature.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively narrow for a topic with substantial prior work in both computer vision and security applications, limiting confidence in the completeness of the gap analysis.",
        "The proposal does not sufficiently justify why COCO, a general-purpose object detection dataset, is an adequate proxy for the weapon-specific fine-tuning dataset it will need to curate.",
        "There is limited discussion of ethical, privacy, and bias considerations inherent to deploying weapon detection in public surveillance settings.",
        "The plan does not specify how class imbalance between rare weapon instances and abundant background frames will be handled during training.",
    ],
    "missing_components": [
        "A concrete data collection and annotation protocol, including sourcing constraints and inter-annotator agreement measures for weapon bounding-box labels.",
        "A discussion of deployment governance, such as human-in-the-loop alert verification, to mitigate the operational risk of false positives.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.80, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to real-time deep learning-based weapon detection in surveillance video, based on the retrieved abstract." for t in titles]
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
