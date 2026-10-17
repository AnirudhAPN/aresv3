import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Early Detection of Diabetic Retinopathy from Fundus Images"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning model that accurately classifies diabetic retinopathy severity from color fundus photographs at an early, pre-symptomatic stage.",
        "Evaluate convolutional and attention-based architectures for their ability to detect subtle microvascular lesions such as microaneurysms and dot hemorrhages.",
        "Assess model generalization across fundus images captured with different camera hardware, populations, and image quality conditions.",
        "Investigate explainability techniques that highlight lesion regions to support clinical trust and adoption of automated screening tools.",
    ],
    "research_questions": [
        "Which deep learning architectures achieve the highest sensitivity for detecting early-stage (mild/moderate) diabetic retinopathy compared to referable-stage disease?",
        "How does image preprocessing (e.g., contrast enhancement, vessel segmentation) affect model performance on low-quality or ungradable fundus images?",
        "To what extent do models trained on one population or camera type generalize to external, unseen datasets?",
        "Can attention-based visualization methods reliably localize clinically relevant lesions to improve interpretability for ophthalmologists?",
    ],
    "keywords": [
        "diabetic retinopathy detection",
        "fundus image classification",
        "convolutional neural networks medical imaging",
        "retinal lesion segmentation",
        "explainable AI ophthalmology",
    ],
    "research_scope": (
        "This research focuses on supervised deep learning approaches for classifying diabetic retinopathy "
        "severity from 2D color fundus photographs, spanning grading tasks from no apparent retinopathy through "
        "proliferative disease, with particular emphasis on early and referable-stage detection. The scope "
        "includes convolutional neural network and vision transformer architectures, transfer learning from "
        "large-scale image corpora, and post-hoc explainability methods such as saliency and attention maps. "
        "It excludes optical coherence tomography (OCT)-based diagnosis, diabetic macular edema quantification "
        "via volumetric imaging, non-deep-learning classical computer vision pipelines, and clinical trial or "
        "cost-effectiveness analyses of screening programs, which are treated as out of scope for this review."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of transfer learning from ImageNet-pretrained CNNs (e.g., ResNet, EfficientNet, Inception) fine-tuned on fundus image datasets to reduce data requirements.",
        "Growing adoption of attention mechanisms and vision transformers to improve localization of small lesions like microaneurysms that CNNs sometimes miss.",
        "Rising interest in explainable AI, particularly Grad-CAM and attention-map visualizations, to build clinician trust in automated grading systems.",
        "Expansion of multi-dataset and federated training strategies to improve cross-population generalization given the demographic imbalance in public fundus datasets.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of model robustness to image quality degradation (blur, uneven illumination, poor pupil dilation) that commonly occurs in real-world community screening settings.",
            "based_on_papers": ["Deep Learning for Detection of Diabetic Eye Disease"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: fine-grained comparison of lesion-level segmentation accuracy (microaneurysms vs. hemorrhages vs. exudates) as a mechanism for explaining severity-grade predictions, rather than relying solely on image-level classification labels.",
            "based_on_papers": ["Automated Grading of Diabetic Retinopathy Using CNNs", "EyeArt: Automated Retinal Screening"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: prospective validation of early-stage detection performance (distinguishing mild non-proliferative disease from no disease) across diverse ethnic and camera-hardware cohorts, as most evaluations emphasize the coarser referable vs. non-referable distinction.",
            "based_on_papers": ["A Deep Learning System for Diabetic Retinopathy Screening"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop lesion-aware multi-task models that jointly predict severity grade and segment individual lesion types to improve early-stage sensitivity and interpretability.",
        "Build standardized, demographically diverse benchmark datasets with quality-graded images to enable fair robustness comparisons across screening conditions.",
        "Integrate uncertainty quantification so that low-confidence predictions on ungradable or borderline images are automatically flagged for human review.",
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
            "description": "Deep residual CNN architecture (ResNet) commonly fine-tuned on fundus images for multi-class diabetic retinopathy severity grading, offering a strong balance of accuracy and training stability.",
            "verified": True,
        },
        {
            "name": "Vision Transformer with Attention Mechanism",
            "url": "https://arxiv.org/abs/2010.11929",
            "description": "Transformer-based architecture that models long-range spatial dependencies across the fundus image, enabling attention maps that localize small lesions such as microaneurysms for improved early-stage detection and interpretability.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Quadratic Weighted Kappa",
        "Sensitivity (Recall) at referable disease threshold",
        "Specificity",
        "Area Under the ROC Curve (AUC)",
    ],
    "implementation_steps": [
        "Collect and consolidate fundus image datasets, ensuring severity labels follow a consistent grading scale such as the International Clinical Diabetic Retinopathy scale",
        "Preprocess images with cropping, contrast-limited adaptive histogram equalization, and quality filtering to remove ungradable photographs",
        "Split data into training, validation, and held-out external test sets stratified by severity grade and, where available, by data source to test generalization",
        "Fine-tune a pretrained CNN or vision transformer backbone using class-balanced loss weighting to address severe class imbalance toward the no-disease category",
        "Apply explainability methods such as Grad-CAM or attention rollout to validate that the model attends to clinically plausible lesion regions",
        "Evaluate on internal and external test sets using quadratic weighted kappa, sensitivity/specificity at clinically relevant operating points, and calibration analysis",
    ],
    "unverified_suggestions": ["EyePACS dataset", "APTOS 2019 Blindness Detection dataset", "MESSIDOR-2 dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly targets an early-detection gap that has meaningful clinical impact, since catching mild non-proliferative disease enables timely intervention before vision loss.",
        "The methodology combines a well-established CNN baseline with a more recent attention-based architecture, allowing direct comparison of interpretability and lesion-localization benefits.",
        "Explainability is treated as a first-class evaluation criterion rather than an afterthought, which is important for eventual clinical adoption.",
        "The evaluation plan appropriately uses quadratic weighted kappa and sensitivity/specificity, metrics that are standard and clinically meaningful for ordinal disease grading tasks.",
    ],
    "weaknesses": [
        "The proposal does not fully specify how class imbalance between early-stage and healthy images will be quantified before choosing a reweighting strategy, leaving evidence sufficiency for the chosen approach unclear.",
        "Dataset fit caveats are underexplored: reliance on commonly used public datasets may not adequately represent the image-quality and demographic variability seen in real-world screening deployments.",
        "There is limited discussion of computational cost or inference latency, which matters for deployment in low-resource screening settings.",
        "The proposal does not address how borderline or ungradable images will be handled at inference time beyond flagging them conceptually.",
    ],
    "missing_components": [
        "A concrete plan for statistical significance testing when comparing the CNN and transformer architectures across metrics.",
        "An ablation study isolating the contribution of the explainability-guided training signal versus standard fine-tuning.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to early-stage diabetic retinopathy detection from fundus images, based on the retrieved abstract." for t in titles]
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
