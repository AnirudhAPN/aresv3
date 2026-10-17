import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Convolutional Neural Networks for Crop Disease Detection in Precision Agriculture"

PLANNER_JSON = {
    "research_objectives": [
        "Develop convolutional neural network architectures that accurately classify crop leaf diseases from field and laboratory-captured images across multiple plant species.",
        "Evaluate the trade-offs between classification accuracy, model size, and inference latency when deploying CNN-based disease detectors on low-cost mobile and edge devices used in precision agriculture.",
        "Investigate transfer learning and data augmentation strategies to improve CNN generalization when labeled disease imagery is scarce for certain crops or regions.",
        "Assess the robustness of CNN disease classifiers to real-world variability such as lighting conditions, occlusion, background clutter, and mixed infections.",
    ],
    "research_questions": [
        "Which CNN architectures (e.g., ResNet, MobileNet, EfficientNet) achieve the best trade-off between diagnostic accuracy and computational efficiency for on-device crop disease detection?",
        "How does transfer learning from large-scale image datasets improve CNN performance on crop disease classification when field-labeled data is limited?",
        "What data augmentation and domain adaptation techniques best close the gap between curated laboratory leaf images and noisy, in-field smartphone captures?",
        "How well do CNN-based disease detectors generalize across unseen crop varieties, geographic regions, and co-occurring diseases not represented in training data?",
    ],
    "keywords": [
        "convolutional neural networks",
        "crop disease detection",
        "precision agriculture",
        "plant leaf image classification",
        "transfer learning agriculture",
    ],
    "research_scope": "This research focuses on convolutional neural network methods for detecting and classifying crop diseases from leaf and canopy imagery within precision agriculture applications. The scope includes supervised CNN architectures, transfer learning from pretrained image models, and data augmentation techniques evaluated on public plant disease datasets as well as field-collected smartphone imagery, with attention to deployment on resource-constrained mobile and edge hardware for real-time farmer-facing diagnostic tools. It excludes non-visual disease detection modalities such as hyperspectral or genomic assays, disease detection in non-crop plants, and purely classical machine learning approaches (e.g., handcrafted feature classifiers) that do not employ deep convolutional representation learning.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread use of transfer learning from ImageNet-pretrained CNNs (ResNet, VGG, MobileNet) fine-tuned on plant disease datasets such as PlantVillage.",
        "Growing adoption of lightweight and mobile-optimized CNN architectures to enable on-device, offline disease diagnosis in low-connectivity rural areas.",
        "Increasing use of data augmentation and synthetic image generation (e.g., GAN-based) to address class imbalance and scarce field-labeled disease imagery.",
        "Rising interest in explainability techniques such as Grad-CAM to help agronomists trust and interpret CNN disease predictions.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: large-scale evaluation of CNN disease classifiers trained on curated laboratory images when deployed on unconstrained, farmer-captured field photos with variable lighting and backgrounds.",
            "based_on_papers": ["Using Deep Learning for Image-Based Plant Disease Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic benchmarking of CNN inference latency, memory footprint, and battery consumption on low-cost smartphones representative of smallholder farmers in developing regions.",
            "based_on_papers": ["Deep Learning for Image-Based Cassava Disease Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparative study of CNN generalization to co-occurring or multiple simultaneous diseases on the same leaf, rather than single-label disease classification.",
            "based_on_papers": ["Using Deep Learning for Image-Based Plant Disease Detection", "Deep Learning for Image-Based Cassava Disease Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop domain adaptation methods that bridge the gap between clean laboratory leaf datasets and noisy, real-world field imagery collected via smartphones.",
        "Design multi-label CNN architectures capable of detecting co-occurring diseases and nutrient deficiencies on a single leaf image.",
        "Build standardized, geographically diverse benchmark datasets with field-captured images and farmer-verified labels to enable fairer cross-study comparison.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "PlantVillage Dataset",
            "url": "https://www.kaggle.com/datasets/emmarex/plantdisease",
            "description": "Labeled crop leaf images for plant disease classification in agriculture.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://arxiv.org/abs/1511.08458",
            "description": "Deep learning architecture that learns hierarchical spatial features from leaf images, forming the core model family for automated crop disease classification.",
            "verified": True,
        },
        {
            "name": "MobileNetV2",
            "url": "https://arxiv.org/abs/1801.04381",
            "description": "Lightweight CNN architecture using depthwise separable convolutions, well-suited for real-time crop disease inference on low-cost mobile and edge devices in the field.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["Accuracy", "Precision", "Recall", "F1-score"],
    "implementation_steps": [
        "Collect and curate crop leaf images from public datasets and supplementary field photography, covering healthy and diseased samples across target crop species.",
        "Preprocess images through resizing, normalization, and augmentation (rotation, flipping, brightness jitter) to improve robustness to field conditions.",
        "Split data into training, validation, and test sets, stratified by disease class and, where possible, by geographic source to test generalization.",
        "Fine-tune pretrained CNN backbones (e.g., MobileNetV2, ResNet) on the disease classification task using transfer learning.",
        "Evaluate trained models on held-out and field-collected test sets using accuracy, precision, recall, and F1-score, with confusion matrix analysis per disease class.",
        "Benchmark model size, inference latency, and energy usage on representative low-cost smartphone hardware to assess field deployability.",
    ],
    "unverified_suggestions": ["PlantDoc Dataset", "Cassava Leaf Disease Classification Dataset", "IP102 Insect Pest Dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly ties CNN architecture choices to real deployment constraints faced by smallholder farmers, such as offline mobile inference.",
        "The methodology combines transfer learning with lightweight architectures, addressing both accuracy and on-device feasibility.",
        "Evaluation metrics and stratified train/test splitting are appropriate for a multi-class, potentially imbalanced disease classification task.",
        "The scope explicitly excludes non-visual and non-crop modalities, keeping the research well-focused and achievable within a reasonable timeframe.",
    ],
    "weaknesses": [
        "The evidence sufficiency for claimed gaps is limited, as it is based on a small retrieved sample of 8 papers rather than a comprehensive systematic review.",
        "The proposal does not fully address dataset fit, since PlantVillage images are largely lab-curated and may not represent the noisy field conditions the deployment targets.",
        "Benchmarking on smartphone hardware is proposed but lacks specificity on which device tiers or chipsets will be tested.",
        "The gap around multi-label or co-occurring disease detection is identified but no concrete multi-label modeling approach is specified in this plan.",
    ],
    "missing_components": [
        "A plan for validating model predictions against expert agronomist ground truth on newly collected field images.",
        "An ablation study isolating the contribution of augmentation strategies versus backbone architecture choice to final accuracy.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.8, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to CNN-based crop disease detection in precision agriculture, based on the retrieved abstract." for t in titles]
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
