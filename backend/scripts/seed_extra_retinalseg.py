import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Retinal Vessel Segmentation in Fundus Photography"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning model that accurately segments retinal blood vessels from color fundus photographs",
        "Improve segmentation robustness to variations in image quality, illumination, and camera type across clinical datasets",
        "Evaluate the diagnostic utility of automated vessel segmentation for early detection of diabetic retinopathy and hypertensive retinopathy",
        "Reduce reliance on large amounts of pixel-level annotated data through semi-supervised or transfer learning strategies",
    ],
    "research_questions": [
        "Which convolutional neural network architectures achieve the best trade-off between segmentation accuracy and computational cost for retinal vessel segmentation?",
        "How well do models trained on one fundus imaging dataset generalize to unseen datasets acquired with different camera equipment?",
        "Can attention mechanisms or multi-scale feature fusion improve segmentation of thin and low-contrast peripheral vessels?",
        "What is the impact of limited annotated training data on segmentation performance, and can data augmentation or transfer learning mitigate this?",
    ],
    "keywords": [
        "retinal vessel segmentation",
        "fundus photography",
        "convolutional neural networks",
        "diabetic retinopathy screening",
        "medical image segmentation",
    ],
    "research_scope": (
        "This research focuses on deep learning approaches for pixel-level segmentation of retinal blood vessels "
        "in color fundus photographs, a foundational task for automated screening of diabetic retinopathy, "
        "hypertensive retinopathy, and other vascular eye diseases. The scope covers convolutional and "
        "encoder-decoder architectures such as U-Net and its variants, evaluated primarily on public benchmark "
        "datasets including DRIVE, STARE, and CHASE_DB1. Emphasis is placed on segmentation accuracy, "
        "cross-dataset generalization, and computational efficiency suitable for deployment in clinical or "
        "point-of-care screening settings, while excluding downstream disease classification tasks that build "
        "on top of the segmented vessel maps."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of U-Net and encoder-decoder architectures with skip connections for vessel segmentation",
        "Increasing use of attention mechanisms and multi-scale feature extraction to capture thin vessel structures",
        "Growing interest in generative adversarial networks for vessel map synthesis and data augmentation",
        "Shift toward evaluating models across multiple public datasets to assess cross-domain generalization",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of vessel segmentation models to low-quality or artifact-laden fundus images captured by low-cost, non-mydriatic cameras used in resource-limited screening programs",
            "based_on_papers": ["Retinal Vessel Segmentation Using Deep Neural Networks", "A Survey on Deep Learning Based Retinal Vessel Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: uncertainty quantification for vessel segmentation predictions to flag low-confidence regions for clinician review",
            "based_on_papers": ["U-Net: Convolutional Networks for Biomedical Image Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized cross-dataset benchmarking protocols that account for differences in vessel annotation guidelines between DRIVE, STARE, and CHASE_DB1",
            "based_on_papers": ["A Survey on Deep Learning Based Retinal Vessel Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop lightweight segmentation architectures optimized for real-time inference on portable fundus cameras",
        "Incorporate self-supervised or semi-supervised pretraining to reduce dependence on scarce pixel-level vessel annotations",
        "Explore domain adaptation techniques to improve model transfer across fundus imaging devices and patient populations",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "PhysioNet", "url": "https://physionet.org/", "description": "Curated clinical and physiological signal datasets for healthcare research.", "verified": True},
        {"name": "DRIVE", "url": "https://drive.grand-challenge.org/", "description": "Digital Retinal Images for Vessel Extraction dataset, containing 40 color fundus photographs with manually annotated vessel segmentations, widely used as a benchmark for retinal vessel segmentation.", "verified": True},
    ],
    "algorithms": [
        {"name": "U-Net", "url": "https://arxiv.org/abs/1505.04597", "description": "Encoder-decoder convolutional neural network with skip connections between corresponding downsampling and upsampling layers, well suited to pixel-level biomedical image segmentation tasks such as vessel extraction.", "verified": True},
        {"name": "ResNet-34 backbone", "url": "https://arxiv.org/abs/1512.03385", "description": "Deep residual convolutional network used as a feature-extracting encoder within segmentation architectures, mitigating vanishing gradients via residual connections to enable deeper, more expressive networks.", "verified": True},
    ],
    "evaluation_metrics": [
        "Sensitivity (recall) for correctly identified vessel pixels",
        "Specificity for correctly identified background pixels",
        "Area under the ROC curve (AUC-ROC)",
        "Dice coefficient (F1 score) measuring overlap between predicted and ground-truth vessel maps",
    ],
    "implementation_steps": [
        "Collect and preprocess fundus images from DRIVE, STARE, and CHASE_DB1, including green-channel extraction and contrast-limited adaptive histogram equalization",
        "Split each dataset into training, validation, and test sets following standard benchmark protocols",
        "Apply data augmentation such as random rotation, flipping, and patch extraction to increase training sample diversity",
        "Train the U-Net (or attention-augmented variant) using a combined Dice and binary cross-entropy loss function",
        "Tune hyperparameters including learning rate, batch size, and patch size using the validation set",
        "Evaluate the trained model on held-out test sets and on cross-dataset splits to assess generalization",
    ],
    "unverified_suggestions": [
        "Consider incorporating vessel topology-aware loss functions to reduce fragmented or disconnected vessel predictions",
        "Explore knowledge distillation to compress large segmentation models for deployment on low-power fundus camera hardware",
        "Investigate federated learning approaches to train across multiple hospital datasets without centralizing sensitive patient images",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research objectives centered on a well-defined medical imaging task with established public benchmarks",
        "Methodology grounded in a proven encoder-decoder architecture (U-Net) with appropriate evaluation metrics for segmentation",
        "Research questions address both technical performance and clinically meaningful generalization concerns",
        "Identified gaps point to concrete, actionable directions such as robustness to low-quality images and uncertainty quantification",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small for a mature subfield with a large existing literature, limiting confidence in the completeness of identified gaps",
        "The proposal does not specify how class imbalance between vessel and background pixels will be addressed during training",
        "No discussion of inter-rater variability in manual vessel annotations used as ground truth, which affects upper-bound performance estimates",
        "Limited consideration of computational resource constraints for deployment in low-resource clinical screening settings",
    ],
    "missing_components": [
        "A clear baseline comparison against classical (non-deep-learning) vessel segmentation methods such as matched filtering",
        "An explicit plan for statistical significance testing when comparing model variants across datasets",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning based retinal vessel segmentation, based on the retrieved abstract." for t in titles]
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
