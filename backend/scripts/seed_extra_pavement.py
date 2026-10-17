import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Convolutional Neural Networks for Automated Pavement Crack Detection in Infrastructure Inspection"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a CNN-based semantic segmentation model that detects and delineates pavement cracks from road surface images captured during infrastructure inspection",
        "Evaluate encoder-decoder segmentation architectures, including U-Net variants and DeepLab, for pixel-level crack detection accuracy under varying lighting, surface texture, and crack severity conditions",
        "Investigate lightweight, efficient CNN architectures suitable for real-time on-vehicle inspection deployment versus larger models optimized purely for accuracy",
        "Assess model robustness to common distractors, such as shadows, lane markings, surface stains, and tar patches, that are frequently confused with cracks in real-world pavement imagery",
    ],
    "research_questions": [
        "Which CNN segmentation architectures achieve the best pixel-level crack detection accuracy while remaining feasible for real-time deployment on inspection vehicles?",
        "How much does detection accuracy degrade under adverse imaging conditions, such as low light, shadows, and wet pavement, compared to controlled daytime conditions?",
        "Can models trained on one road surface type or region generalize to pavement imagery from different regions or surface materials without retraining?",
        "What is the false-positive rate of crack detectors on common visual distractors, such as shadows, lane markings, and tar patches, and which architectural or augmentation strategies most reduce it?",
    ],
    "keywords": [
        "pavement crack detection deep learning",
        "road surface segmentation CNN",
        "infrastructure inspection computer vision",
        "crack semantic segmentation",
        "automated pavement distress detection",
    ],
    "research_scope": (
        "This research covers CNN-based semantic segmentation and detection models for identifying and "
        "delineating cracks in road pavement surface images captured by inspection vehicles or handheld "
        "cameras, including real-time and efficient architecture considerations and robustness to visual "
        "distractors. The scope excludes non-crack pavement distresses such as potholes and rutting as a "
        "primary focus, subsurface inspection via ground-penetrating radar, and structural crack detection in "
        "bridges or buildings outside pavement contexts."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread use of U-Net and DeepLab-family encoder-decoder architectures for pixel-level crack segmentation",
        "Growing interest in lightweight and mobile CNN backbones, such as MobileNet and EfficientNet, to enable real-time on-vehicle inspection processing",
        "Increasing use of synthetic and augmented crack imagery, including simulated shadows and lighting variation, to improve robustness to real-world distractors",
        "Adoption of attention mechanisms and multi-scale feature fusion to better capture thin, low-contrast crack structures against textured pavement backgrounds",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of segmentation accuracy versus inference speed tradeoffs across lightweight and full-size CNN architectures under a shared crack detection benchmark.",
            "based_on_papers": ["U-Net: Convolutional Networks for Biomedical Image Segmentation", "DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-region and cross-surface-material generalization testing, since most models are trained and evaluated on pavement imagery from a single geographic or material context.",
            "based_on_papers": ["DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: consistent reporting of false-positive rates on common visual distractors such as shadows, tar patches, and lane markings, which makes real-world deployment reliability difficult to assess.",
            "based_on_papers": ["U-Net: Convolutional Networks for Biomedical Image Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build multi-region, multi-surface-material crack detection benchmarks to systematically test generalization",
        "Establish standardized distractor robustness test sets and report false-positive rates alongside segmentation accuracy",
        "Explore knowledge distillation from high-accuracy segmentation models into lightweight architectures for real-time on-vehicle deployment",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "U-Net",
            "url": "https://arxiv.org/abs/1505.04597",
            "description": "Encoder-decoder segmentation architecture with skip connections, used as the primary backbone for pixel-level crack mask prediction from pavement images.",
            "verified": True,
        },
        {
            "name": "DeepLabv3+",
            "url": "https://arxiv.org/abs/1802.02611",
            "description": "Atrous-convolution segmentation architecture with multi-scale context aggregation, used as a comparison architecture for capturing thin, elongated crack structures.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Intersection over Union (IoU)",
        "Pixel-level F1-score / Dice coefficient",
        "Precision-recall on crack pixels",
        "Inference speed (frames per second)",
    ],
    "implementation_steps": [
        "Assemble labeled pavement crack imagery spanning multiple regions, surface materials, and lighting conditions, with pixel-level annotation masks",
        "Preprocess images with consistent resolution, normalization, and augmentation, including simulated shadows and lighting variation for robustness training",
        "Implement and train U-Net and DeepLabv3+ segmentation models, along with a lightweight backbone variant (e.g., MobileNet-based encoder) for efficiency comparison",
        "Curate a dedicated distractor evaluation set containing shadows, lane markings, tar patches, and stains to measure false-positive rates explicitly",
        "Evaluate segmentation accuracy (IoU, Dice, precision-recall) and inference speed (FPS) across all architecture variants on held-out test imagery",
        "Test cross-region and cross-surface-material generalization by training on one region/material and evaluating zero-shot on a different one",
    ],
    "unverified_suggestions": ["CRACK500", "CFD (Crack Forest Dataset)", "DeepCrack dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly addresses the practical accuracy-versus-speed tradeoff central to real-time on-vehicle infrastructure inspection deployment",
        "A dedicated distractor evaluation set for shadows, lane markings, and tar patches provides a concrete, actionable measure of real-world false-positive risk",
        "The methodology includes cross-region and cross-surface-material generalization testing, addressing a common and consequential blind spot in prior crack detection studies",
        "Evaluation metrics (IoU, Dice, precision-recall, FPS) appropriately balance segmentation quality with deployment feasibility",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how crack severity or width will be stratified during evaluation, despite hairline versus wide cracks posing substantially different detection difficulty",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host pavement crack image benchmarks such as CRACK500 or the Crack Forest Dataset, so those remain unverified suggestions requiring separate access",
        "The plan does not address how annotation consistency for thin, low-contrast crack boundaries will be verified across multiple human annotators, which is a known source of label noise in crack segmentation datasets",
    ],
    "missing_components": [
        "A crack-severity or width-stratified evaluation protocol distinguishing hairline from wide/structural cracks",
        "An inter-annotator agreement analysis for pixel-level crack boundary labeling to assess ground-truth label quality",
    ],
    "novelty_score": 0.60,
    "quality_score": 0.70,
}

CONFIDENCE_JSON = {"internal_consistency": 0.75, "specificity": 0.78, "completeness": 0.77}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to CNN-based automated pavement crack detection, based on the retrieved abstract." for t in titles]
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
