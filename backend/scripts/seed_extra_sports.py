import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Pose Estimation Networks for Injury Prevention in Sports Analytics"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate the accuracy of deep learning-based pose estimation networks for tracking athlete biomechanics in real-world sporting environments.",
        "Identify joint kinematic patterns and movement asymmetries that correlate with elevated injury risk during high-intensity athletic movements.",
        "Design a pose-estimation-driven analytics pipeline that flags at-risk movement signatures in near real time during training sessions.",
        "Assess the generalizability of existing pose estimation models across different sports, camera setups, and athlete body types.",
    ],
    "research_questions": [
        "How accurately can 2D and 3D pose estimation networks recover joint angles relevant to common sports injuries (e.g., ACL tears, hamstring strains)?",
        "Which kinematic features extracted from pose estimation output are most predictive of non-contact injury risk?",
        "Can multi-camera or monocular pose estimation systems be deployed cost-effectively for continuous injury-risk monitoring in team sports settings?",
        "What are the limitations of current pose estimation architectures when applied to fast, occlusion-heavy sporting movements?",
    ],
    "keywords": [
        "human pose estimation",
        "sports biomechanics",
        "injury risk prediction",
        "markerless motion capture",
        "athlete movement analytics",
    ],
    "research_scope": (
        "This research focuses on the application of deep learning-based human pose estimation networks "
        "(2D and 3D) to injury prevention in sports analytics, covering markerless motion capture pipelines, "
        "joint kinematic feature extraction, and downstream injury-risk modeling. The scope spans model "
        "architectures (e.g., CNN-based heatmap regression, transformer-based pose estimators), benchmark "
        "datasets used for pretraining and evaluation, and practical deployment considerations such as "
        "occlusion handling, camera placement, and real-time inference constraints in field and lab settings. "
        "It excludes wearable-sensor-only approaches and focuses primarily on vision-based methods."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of transformer-based pose estimation architectures (e.g., ViTPose) over purely convolutional heatmap regression models for improved accuracy on occluded joints.",
        "Growing use of markerless 3D pose estimation from multi-view or monocular video as a low-cost alternative to traditional marker-based motion capture in sports science labs.",
        "Rising interest in combining pose-derived kinematic features with machine learning classifiers to predict non-contact injury risk (e.g., ACL, hamstring) rather than relying on pose estimation alone.",
        "Emergence of real-time, edge-deployable lightweight pose estimation models aimed at on-field monitoring during live training and competition.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized validation of pose estimation joint-angle accuracy against gold-standard marker-based motion capture specifically for sport-specific high-velocity movements such as cutting and jumping.",
            "based_on_papers": ["Deep High-Resolution Representation Learning for Human Pose Estimation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: longitudinal studies linking pose-estimation-derived kinematic markers to actual prospective injury incidence rather than retrospective or cross-sectional risk scoring.",
            "based_on_papers": ["Stacked Hourglass Networks for Human Pose Estimation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness evaluation of pose estimation models under sports-specific occlusion conditions, such as player-on-player contact and fast multi-athlete scenes in team sports.",
            "based_on_papers": ["Simple Baselines for Human Pose Estimation and Tracking"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop domain-specific pose estimation benchmarks annotated with clinically validated injury-risk kinematic thresholds (e.g., knee valgus angle, trunk lean).",
        "Combine pose estimation time-series output with sequence models (e.g., temporal transformers) to predict injury risk trajectories across a training season.",
        "Investigate federated or on-device lightweight pose estimation models to enable privacy-preserving, low-latency injury-risk monitoring at scale across youth and amateur sports.",
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
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "description": "Deep learning architecture that uses convolutional layers to extract spatial features from images, forming the backbone of most heatmap-based pose estimation models.",
            "verified": True,
        },
        {
            "name": "High-Resolution Network (HRNet)",
            "url": "https://arxiv.org/abs/1902.09212",
            "description": "Pose estimation architecture that maintains high-resolution feature representations throughout the network, improving accuracy for precise joint localization needed in biomechanical analysis.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Percentage of Correct Keypoints (PCK)",
        "Mean Per Joint Position Error (MPJPE)",
        "Object Keypoint Similarity (OKS) / mean Average Precision (mAP)",
        "Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for injury-risk classification",
    ],
    "implementation_steps": [
        "Collect and preprocess multi-view or monocular video footage of athletes performing sport-specific movements (e.g., cutting, landing, sprinting)",
        "Fine-tune a pretrained pose estimation backbone (e.g., HRNet) on a sport-specific annotated dataset to improve joint localization accuracy under occlusion",
        "Extract per-frame joint angles and derived kinematic features such as knee valgus angle, hip-knee flexion ratio, and trunk lean",
        "Aggregate kinematic features into time-series representations and train a supervised injury-risk classifier using labeled prospective injury outcomes",
        "Validate joint-angle estimates against marker-based motion capture ground truth on a held-out calibration cohort",
        "Deploy the pipeline in a near-real-time monitoring setup and evaluate inference latency and robustness under field conditions",
    ],
    "unverified_suggestions": [
        "Explore synthetic data augmentation using simulated athlete motion to expand training data for underrepresented injury-prone movement patterns.",
        "Investigate transfer learning from general human pose datasets to sport-specific domains with limited labeled injury data.",
        "Consider multi-modal fusion of pose estimation output with wearable IMU data to improve injury-risk prediction robustness.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly ties pose estimation methodology to a concrete, high-impact application domain (sports injury prevention) rather than treating pose estimation as an end in itself.",
        "The methodology section specifies both a well-established backbone architecture and quantitative evaluation metrics appropriate for biomechanical validation.",
        "The gap analysis identifies a coherent, actionable research direction around longitudinal injury outcome validation rather than restating known limitations.",
        "The research scope appropriately bounds the investigation to vision-based methods, making the proposal tractable within a single research effort.",
    ],
    "weaknesses": [
        "The proposal does not sufficiently justify why the retrieved evidence base is adequate to support claims about injury-risk prediction, since most cited work concerns generic pose estimation accuracy rather than injury outcomes.",
        "There is limited discussion of class imbalance, given that injury events are rare relative to total training observations.",
        "The proposal does not address ethical or privacy considerations of continuous athlete video monitoring.",
        "Dataset fit is questionable: COCO is a generic object/keypoint dataset not tailored to high-velocity sports movements, and the proposal does not identify a sport-specific alternative or supplementary dataset.",
    ],
    "missing_components": [
        "A clear baseline comparison against traditional marker-based motion capture or existing clinical injury-risk screening tools.",
        "A discussion of statistical power and sample size requirements for the proposed injury-risk classifier given the rarity of injury events.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to pose estimation for sports injury risk analysis, based on the retrieved abstract." for t in titles]
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
