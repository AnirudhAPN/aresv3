import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Convolutional Neural Networks for Automated Satellite Image Cloud Detection"

PLANNER_JSON = {
    "research_objectives": [
        "Develop CNN-based semantic segmentation models that classify cloud, cloud shadow, and clear-sky pixels in multispectral satellite imagery",
        "Evaluate encoder-decoder architectures such as U-Net against classical radiometric threshold algorithms (e.g., Fmask) for cloud mask accuracy on Landsat 8 and Sentinel-2 imagery",
        "Investigate the impact of multispectral band selection (visible, near-infrared, shortwave-infrared) on cloud detection accuracy compared to RGB-only inputs",
        "Assess cross-sensor generalization of CNN cloud detectors trained on one satellite platform when applied to imagery from a different sensor or resolution",
    ],
    "research_questions": [
        "Do CNN-based semantic segmentation architectures outperform classical radiometric threshold algorithms for pixel-level cloud and cloud-shadow detection?",
        "Which multispectral bands contribute most to cloud detection accuracy, and how much does performance degrade when restricted to RGB-only imagery?",
        "How well do CNN cloud detectors trained on Landsat 8 imagery generalize to Sentinel-2 imagery with different spatial resolution and band configurations?",
        "What is the impact of thin cirrus clouds and cloud-shadow ambiguity on segmentation accuracy compared to detection of thick, opaque cloud cover?",
    ],
    "keywords": [
        "cloud detection satellite imagery",
        "semantic segmentation remote sensing",
        "U-Net cloud masking",
        "multispectral image classification",
        "Landsat Sentinel-2 cloud cover",
    ],
    "research_scope": (
        "This research focuses on convolutional neural network methods for pixel-level cloud and cloud-shadow "
        "detection in multispectral optical satellite imagery from platforms such as Landsat 8 and Sentinel-2. The "
        "scope covers encoder-decoder segmentation architectures, multispectral band selection studies, and "
        "cross-sensor generalization evaluation, benchmarked against classical radiometric threshold cloud-masking "
        "algorithms. It excludes atmospheric correction algorithms applied after cloud masking, synthetic aperture "
        "radar (SAR) imagery that is largely unaffected by cloud cover, and weather forecasting applications, which "
        "are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of U-Net and other encoder-decoder CNN architectures for pixel-level cloud segmentation, surpassing patch-level classification approaches",
        "Continued use of classical radiometric threshold algorithms such as Fmask as a baseline and, in some pipelines, as a source of weak labels for training CNNs",
        "Increasing use of multispectral and shortwave-infrared bands beyond RGB to improve discrimination between clouds, snow, and bright surfaces",
        "Rising interest in cross-sensor generalization, motivated by the practical need to apply models trained on one satellite constellation to imagery from another",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of cross-sensor generalization when a CNN cloud detector trained on Landsat 8 imagery is applied directly to Sentinel-2 imagery without fine-tuning.",
            "based_on_papers": ["Cloud-Net: An End-to-end Cloud Detection Algorithm for Landsat 8 Imagery"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative breakdown of segmentation accuracy specifically for thin cirrus clouds and ambiguous cloud-shadow boundaries, as distinct from overall pixel accuracy dominated by easy, opaque cloud cases.",
            "based_on_papers": ["U-Net: Convolutional Networks for Biomedical Image Segmentation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: ablation isolating the marginal contribution of individual non-RGB spectral bands (e.g., shortwave-infrared) to cloud detection accuracy, rather than reporting only full-band-stack results.",
            "based_on_papers": ["Cloud-Net: An End-to-end Cloud Detection Algorithm for Landsat 8 Imagery"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop domain-adaptation techniques that let CNN cloud detectors transfer across sensors with different band configurations and spatial resolutions without full retraining",
        "Create benchmark subsets specifically labeled for thin cirrus and shadow-ambiguous regions to enable fine-grained evaluation beyond aggregate pixel accuracy",
        "Conduct systematic band-ablation studies to identify minimal spectral band subsets that preserve cloud detection accuracy for cost-constrained sensor designs",
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
            "description": "Encoder-decoder convolutional architecture with skip connections, used as the core semantic segmentation backbone for pixel-level cloud and cloud-shadow masking.",
            "verified": True,
        },
        {
            "name": "Cloud-Net",
            "url": "https://arxiv.org/abs/1901.10132",
            "description": "CNN architecture purpose-built for end-to-end cloud detection on Landsat 8 imagery, used as a domain-specific comparison point against the general-purpose U-Net baseline.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Pixel accuracy",
        "Intersection over Union (IoU) for cloud and shadow classes",
        "Precision and Recall per class (cloud, shadow, clear sky)",
        "F1-score on thin-cirrus and shadow-ambiguous subsets",
    ],
    "implementation_steps": [
        "Assemble multispectral image tiles from Landsat 8 and Sentinel-2 archives with corresponding cloud/shadow ground-truth masks",
        "Preprocess tiles into fixed-size patches, normalizing reflectance values per band and handling differing spatial resolutions across sensors",
        "Train a U-Net segmentation model using the full multispectral band stack, and a second variant restricted to RGB-only bands for comparison",
        "Train Cloud-Net as a domain-specific architecture baseline on the same Landsat 8 training split",
        "Evaluate all models against the classical Fmask threshold algorithm on held-out tiles, reporting per-class IoU and F1-score, with separate breakdowns for thin-cirrus and shadow-ambiguous regions",
        "Test cross-sensor generalization by applying the Landsat-trained models directly to Sentinel-2 test tiles without additional fine-tuning",
    ],
    "unverified_suggestions": [
        "38-Cloud: A Cloud Segmentation Dataset for Landsat 8 Imagery",
        "Sentinel-2 Cloud Mask Catalogue",
        "L8 Biome Cloud Validation Masks dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal benchmarks against a well-established classical baseline (Fmask), giving a meaningful reference point rather than only comparing among deep learning variants",
        "Including a dedicated evaluation breakdown for thin cirrus and shadow-ambiguous regions targets the genuinely hard cases rather than relying on aggregate pixel accuracy that easy cases dominate",
        "The band-ablation comparison (full multispectral stack versus RGB-only) directly tests a practically relevant question for lower-cost sensor designs",
        "Testing cross-sensor generalization from Landsat 8 to Sentinel-2 without fine-tuning is a realistic and informative stress test of model robustness",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how ground-truth cloud masks will be reconciled between Landsat 8 and Sentinel-2, which use different reference labeling conventions and resolutions",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host multispectral satellite cloud-masking datasets, so the project depends on the unverified 38-Cloud and Sentinel-2 Cloud Mask Catalogue suggestions being accessible and sufficiently annotated",
        "The plan does not address seasonal and geographic bias in the training data (e.g., underrepresentation of polar snow/cloud confusion cases), which is a known failure mode for cloud detectors",
    ],
    "missing_components": [
        "A discussion of snow and bright-surface confusion cases, which are a well-documented source of false positives in cloud detection",
        "A plan for fine-tuning or lightweight domain adaptation if zero-shot cross-sensor transfer underperforms, rather than treating it as a binary pass/fail test",
    ],
    "novelty_score": 0.60,
    "quality_score": 0.72,
}

CONFIDENCE_JSON = {"internal_consistency": 0.80, "specificity": 0.77, "completeness": 0.81}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to CNN-based automated satellite image cloud detection, based on the retrieved abstract." for t in titles]
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
