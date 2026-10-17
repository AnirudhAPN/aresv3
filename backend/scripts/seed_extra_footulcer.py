import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Grading of Diabetic Foot Ulcer Severity"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning pipeline that automatically grades diabetic foot ulcer (DFU) severity from clinical wound images",
        "Evaluate convolutional and transformer-based architectures for their ability to classify DFU severity according to established clinical grading systems (e.g., Wagner, University of Texas)",
        "Assess the clinical reliability and generalizability of automated grading models across diverse patient populations and imaging conditions",
        "Investigate explainability techniques that help clinicians trust and interpret automated DFU severity predictions",
    ],
    "research_questions": [
        "Can deep learning models match or exceed expert clinician agreement when grading diabetic foot ulcer severity from photographs?",
        "Which model architectures and training strategies best handle the class imbalance and limited labeled data typical of DFU image datasets?",
        "How well do automated grading models generalize to images captured with different smartphone cameras and lighting conditions in real-world clinical settings?",
        "What explainability methods most effectively communicate model reasoning for DFU severity grades to clinicians and patients?",
    ],
    "keywords": [
        "diabetic foot ulcer classification",
        "wound severity grading",
        "convolutional neural networks for dermatology",
        "medical image segmentation",
        "explainable AI in wound care",
    ],
    "research_scope": "This research focuses on the development and evaluation of deep learning models for automated grading of diabetic foot ulcer (DFU) severity using clinical wound photographs. The scope encompasses image preprocessing and segmentation to isolate the wound region, classification of severity according to standardized clinical scales such as the Wagner and University of Texas systems, and validation against expert clinician assessments. The work also considers deployment constraints relevant to point-of-care and telemedicine settings, including model robustness to variable image capture conditions and the need for interpretable predictions that support clinical decision-making rather than replace clinician judgment.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of convolutional neural networks (CNNs), particularly ResNet and EfficientNet variants, for binary DFU detection and infection/ischemia classification",
        "Growing adoption of U-Net and its variants for wound boundary segmentation prior to severity classification",
        "Emergence of transformer-based vision models applied to dermatological and wound imaging tasks",
        "Rising interest in mobile and telemedicine-deployed DFU screening tools using smartphone-captured images",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: multi-class severity grading that jointly models wound depth, infection, and ischemia according to a unified clinical scale such as the full Wagner or UT classification, rather than treating these as separate binary tasks",
            "based_on_papers": ["Automated identification of diabetic foot ulcer using convolutional neural networks", "DFU severity classification using deep learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of model performance across skin tones, wound locations, and image acquisition devices, which limits confidence in real-world generalizability",
            "based_on_papers": ["Diabetic foot ulcer grading using deep convolutional neural networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: longitudinal validation that tracks whether automated severity grades correlate with actual healing trajectories or amputation risk over time, as opposed to single-timepoint classification accuracy",
            "based_on_papers": ["Deep learning-based diabetic foot ulcer classification for telemedicine"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop multi-task architectures that jointly predict wound depth, infection status, and ischemia to produce a composite clinical severity grade",
        "Curate and release demographically diverse, multi-device DFU image datasets with standardized annotation protocols to enable fair benchmarking",
        "Design prospective clinical studies that link automated severity grades to longitudinal outcomes such as healing time and amputation risk",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "PhysioNet", "url": "https://physionet.org/", "description": "Curated clinical and physiological signal datasets for healthcare research.", "verified": True},
        {"name": "DFUC Grand Challenge Dataset", "url": "https://dfu-challenge.github.io/", "description": "Publicly available diabetic foot ulcer image dataset with expert-annotated bounding boxes and severity labels, used in the DFUC2020/2021/2022 challenges.", "verified": True},
    ],
    "algorithms": [
        {"name": "Convolutional Neural Network (CNN)", "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network", "description": "Used as the backbone classifier (e.g., ResNet50 or EfficientNet-B3) to extract hierarchical visual features from wound images and predict severity class.", "verified": True},
        {"name": "U-Net", "url": "https://arxiv.org/abs/1505.04597", "description": "Encoder-decoder segmentation architecture used to isolate the wound boundary from surrounding healthy tissue prior to severity classification, improving feature localization.", "verified": True},
    ],
    "evaluation_metrics": [
        "Macro-averaged F1-score across severity classes",
        "Cohen's kappa for agreement with expert clinician grading",
        "Sensitivity and specificity for detecting high-severity (limb-threatening) ulcers",
        "Intersection-over-Union (IoU) for wound segmentation accuracy",
    ],
    "implementation_steps": [
        "Collect and consolidate DFU image data from public datasets such as the DFUC Grand Challenge, applying consistent severity annotations aligned to the Wagner or UT scale",
        "Preprocess images with standardized resizing, color normalization, and augmentation (rotation, flipping, brightness jitter) to improve robustness to capture conditions",
        "Train a U-Net segmentation model to localize the wound region and crop images to this region of interest",
        "Fine-tune a CNN classifier (e.g., EfficientNet-B3) on the segmented wound crops to predict discrete severity grades",
        "Validate the trained pipeline using stratified k-fold cross-validation and compare predictions against expert clinician grades using Cohen's kappa",
        "Apply an explainability method such as Grad-CAM to visualize the image regions driving each severity prediction for clinician review",
    ],
    "unverified_suggestions": [
        "Explore transformer-based vision architectures (e.g., Vision Transformer or Swin Transformer) as an alternative to CNN backbones for severity classification",
        "Investigate federated learning approaches to enable multi-hospital model training without centralizing sensitive patient wound images",
        "Consider incorporating thermal or multispectral imaging alongside standard RGB photographs to improve detection of early ischemia",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly grounds severity grading in established clinical classification systems (Wagner, University of Texas), improving clinical relevance",
        "The two-stage segmentation-then-classification pipeline is a sound and well-precedented design choice for wound image analysis",
        "Evaluation metrics appropriately emphasize agreement with clinician judgment (Cohen's kappa) rather than raw accuracy alone",
        "The inclusion of explainability (Grad-CAM) addresses a real clinical adoption barrier for automated diagnostic tools",
    ],
    "weaknesses": [
        "The proposal does not specify how class imbalance between mild and severe/limb-threatening ulcer cases will be handled during training",
        "There is limited discussion of how the proposed dataset (DFUC) matches the diversity of skin tones and wound presentations needed for the generalizability claims made in the research questions",
        "No baseline comparison against existing published DFU classification models is specified, making it hard to contextualize expected performance gains",
        "The evidence supporting the future direction on longitudinal outcome validation is thin, as none of the retrieved papers provide longitudinal healing data",
    ],
    "missing_components": [
        "A concrete plan for obtaining institutional review board (IRB) approval and patient consent if any prospective or hospital-sourced images are used",
        "A defined baseline model or prior published benchmark against which the proposed pipeline's performance will be compared",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning-based diabetic foot ulcer severity grading, based on the retrieved abstract." for t in titles]
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
