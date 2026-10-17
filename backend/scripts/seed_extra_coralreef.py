import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Convolutional Neural Networks for Automated Coral Reef Health Assessment from Underwater Imagery"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a convolutional neural network pipeline capable of classifying coral reef health states from underwater photographic and video imagery",
        "Evaluate the effectiveness of CNN-based semantic segmentation for distinguishing live coral, bleached coral, algae overgrowth, and bare substrate",
        "Assess model generalization across different reef sites, camera systems, and water turbidity conditions",
        "Compare automated CNN-based health assessment against traditional manual reef survey methods in terms of accuracy and throughput",
    ],
    "research_questions": [
        "Can CNN architectures reliably distinguish between healthy, bleached, and diseased coral from underwater RGB imagery?",
        "How does image quality degradation (turbidity, lighting variation, motion blur) affect CNN classification accuracy in underwater environments?",
        "What is the comparative performance of classification versus segmentation approaches for reef health mapping at the colony level?",
        "How well do models trained on one reef region transfer to visually distinct reef ecosystems in other geographic locations?",
    ],
    "keywords": [
        "coral reef health classification",
        "underwater image segmentation",
        "convolutional neural networks",
        "coral bleaching detection",
        "marine ecological monitoring",
    ],
    "research_scope": "This research focuses on the design, training, and evaluation of convolutional neural network models for automated assessment of coral reef health using underwater photographic and video imagery. The scope encompasses image preprocessing techniques to correct for underwater color attenuation and turbidity, CNN and encoder-decoder segmentation architectures for pixel-level coral classification, and benchmarking against annotated reef survey datasets. The work considers multiple coral health categories including healthy, bleached, algae-covered, and diseased states, and evaluates cross-site generalization to determine practical deployment viability for reef monitoring programs, while excluding non-visual sensing modalities such as acoustic or chemical monitoring.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of transfer learning from ImageNet-pretrained CNNs (ResNet, VGG, EfficientNet) fine-tuned on small labeled coral image datasets",
        "Growing use of semantic segmentation architectures such as U-Net and DeepLab for pixel-level coral cover estimation rather than whole-image classification",
        "Integration of underwater image enhancement preprocessing (color correction, dehazing) as a standard step before CNN inference to counteract turbidity and light attenuation",
        "Shift toward citizen-science and autonomous underwater vehicle (AUV) image collection pipelines feeding directly into automated classification systems",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized cross-region benchmark datasets that would allow fair comparison of model generalization across geographically distinct reef ecosystems.",
            "based_on_papers": ["Deep learning for coral reef image classification: a review", "Automated coral cover estimation using convolutional neural networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantification of how CNN classification accuracy degrades under varying turbidity and lighting conditions typical of real-world monitoring dives rather than curated benchmark images.",
            "based_on_papers": ["Underwater image enhancement for coral reef monitoring applications"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparative cost-benefit analysis of CNN-based automated assessment versus expert manual reef surveys in terms of labor hours saved and detection accuracy for early-stage bleaching events.",
            "based_on_papers": ["Semantic segmentation of coral reef benthic imagery"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Development of large-scale, multi-region, expert-annotated coral health datasets to serve as standardized generalization benchmarks",
        "Investigation of domain adaptation and self-supervised pretraining techniques to reduce reliance on scarce labeled underwater imagery",
        "Deployment of lightweight CNN variants optimized for on-device inference on AUVs and diver-carried cameras for real-time reef health feedback",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "Copernicus Open Access Hub", "url": "https://scihub.copernicus.eu/", "description": "Satellite earth-observation data, commonly used for climate research.", "verified": True},
    ],
    "algorithms": [
        {"name": "Convolutional Neural Network (CNN)", "url": "https://www.tensorflow.org/tutorials/images/cnn", "description": "A deep learning architecture that applies learned convolutional filters to extract spatial features from underwater images, used here as the backbone for classifying coral reef health categories such as healthy, bleached, and diseased coral.", "verified": True},
        {"name": "U-Net Segmentation Architecture", "url": "https://arxiv.org/abs/1505.04597", "description": "An encoder-decoder convolutional architecture with skip connections originally designed for biomedical image segmentation, adapted here to produce pixel-level coral cover maps distinguishing live coral, algae, and bare substrate in reef imagery.", "verified": True},
    ],
    "evaluation_metrics": [
        "Overall classification accuracy across coral health categories",
        "Mean Intersection over Union (mIoU) for segmentation of coral cover classes",
        "Per-class F1-score, particularly for the bleached and diseased minority classes",
        "Cohen's kappa agreement between model predictions and expert reef surveyor annotations",
    ],
    "implementation_steps": [
        "Collect and curate underwater imagery from multiple reef sites, ensuring coverage of healthy, bleached, algae-covered, and diseased coral states",
        "Apply underwater image enhancement preprocessing to correct for color attenuation, turbidity, and non-uniform lighting",
        "Annotate images at the colony or pixel level using expert-validated labels for coral health categories",
        "Train and fine-tune CNN classification and U-Net segmentation models using transfer learning from ImageNet-pretrained backbones",
        "Evaluate model performance on held-out imagery from a geographically distinct reef site to test cross-region generalization",
        "Benchmark automated predictions against manual expert reef survey results to assess practical deployment accuracy and throughput gains",
    ],
    "unverified_suggestions": [
        "Consider incorporating hyperspectral or multispectral imagery alongside RGB to improve discrimination between visually similar coral stress states",
        "Explore active learning strategies to prioritize expert annotation effort on the most uncertain or ambiguous coral images",
        "Investigate temporal video-based models that track colony-level health changes across repeated monitoring dives rather than single still frames",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly defined research questions that connect CNN architecture choices to specific coral health assessment outcomes",
        "Methodology appropriately pairs classification and segmentation approaches to address both scene-level and colony-level assessment needs",
        "Evaluation metrics are well-suited to the class-imbalance challenges typical of bleaching and disease detection tasks",
        "Research scope explicitly accounts for real-world underwater imaging challenges such as turbidity and lighting variation",
    ],
    "weaknesses": [
        "The proposal does not specify how class imbalance between healthy and rare disease categories will be handled during training",
        "Insufficient detail on the size and diversity of the annotated training dataset needed to support the proposed cross-region generalization claims",
        "The evidence base drawn from the retrieved literature is relatively sparse for underwater-specific augmentation techniques, so recommendations may not generalize well to all camera setups",
        "No discussion of how the chosen datasets and reviewed studies match the specific imaging conditions (depth, turbidity, camera type) of the target deployment reefs, raising a potential dataset-fit concern",
    ],
    "missing_components": [
        "A concrete data annotation protocol describing inter-annotator agreement procedures among reef ecology experts",
        "A plan for handling temporal and seasonal variation in coral appearance (e.g., bleaching progression over time) within the training data",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to CNN-based coral reef health assessment from underwater imagery, based on the retrieved abstract." for t in titles]
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
