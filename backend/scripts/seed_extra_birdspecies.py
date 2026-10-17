import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Bird Species Identification from Camera Trap Images"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning classifiers that identify bird species from camera trap images captured under variable lighting, occlusion, and pose conditions",
        "Compare convolutional neural network and vision transformer architectures for fine-grained bird species classification accuracy on camera trap imagery",
        "Investigate the effect of class imbalance and long-tailed species distributions, typical of camera trap datasets, on model performance",
        "Assess model robustness to common camera trap artifacts such as motion blur, partial occlusion by vegetation, and nighttime infrared imagery",
    ],
    "research_questions": [
        "Do vision transformer architectures outperform convolutional neural networks for fine-grained bird species classification on camera trap images?",
        "How does classification accuracy degrade for rare species in long-tailed camera trap datasets, and which rebalancing strategies mitigate this?",
        "What is the impact of camera trap-specific artifacts (motion blur, partial occlusion, infrared/nighttime capture) on classification accuracy?",
        "Can models trained on camera trap data from one geographic region generalize to camera trap deployments in a different region with overlapping species?",
    ],
    "keywords": [
        "camera trap image classification",
        "bird species recognition",
        "fine-grained visual classification",
        "wildlife monitoring deep learning",
        "convolutional neural network species identification",
    ],
    "research_scope": (
        "This research covers deep learning approaches for classifying bird species from still images captured by "
        "camera traps, including convolutional and transformer-based architectures, class-imbalance mitigation, and "
        "robustness to camera trap-specific artifacts such as motion blur, occlusion, and infrared capture. It "
        "excludes video-based or audio-based bird identification, species detection tasks focused purely on presence "
        "versus absence without species-level labels, and non-avian wildlife classification."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of vision transformer backbones alongside convolutional networks for fine-grained wildlife species classification",
        "Growing adoption of long-tailed recognition techniques (class-balanced loss, re-sampling) to address the highly imbalanced species distributions found in camera trap datasets",
        "Use of transfer learning from large-scale citizen-science image datasets to bootstrap camera trap classifiers with limited labeled data",
        "Emerging interest in domain adaptation methods to handle infrared/nighttime imagery and site-specific visual conditions",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of vision transformer and convolutional backbones specifically on camera trap bird imagery, as most fine-grained classification benchmarks use curated, non-camera-trap photographs.",
            "based_on_papers": ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of cross-region generalization for camera trap bird classifiers trained on one geographic deployment and tested on another.",
            "based_on_papers": ["Deep Learning for Camera Trap Image Classification: A Review"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantified accuracy breakdown by specific camera-trap artifact type (motion blur, occlusion, infrared), since most studies report only aggregate accuracy.",
            "based_on_papers": ["Deep Learning for Camera Trap Image Classification: A Review"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build cross-region camera trap benchmarks with overlapping species to systematically test model generalization",
        "Develop artifact-stratified evaluation protocols that report accuracy separately for blurred, occluded, and infrared images",
        "Explore domain adaptation and few-shot fine-tuning techniques for rapidly onboarding new camera trap deployment sites with minimal labeled data",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "ResNet",
            "url": "https://arxiv.org/abs/1512.03385",
            "description": "Deep residual convolutional network used as a baseline backbone for fine-grained bird species classification from camera trap images.",
            "verified": True,
        },
        {
            "name": "Vision Transformer (ViT)",
            "url": "https://arxiv.org/abs/2010.11929",
            "description": "Patch-based transformer image classifier evaluated against convolutional backbones for its ability to capture fine-grained plumage and pose distinctions relevant to species identification.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Top-1 and top-5 classification accuracy",
        "Per-class (species) F1-score, reported separately for head and tail classes",
        "Balanced accuracy under long-tailed class distribution",
        "Accuracy stratified by artifact type (blur, occlusion, infrared)",
    ],
    "implementation_steps": [
        "Assemble and clean a labeled camera trap bird image dataset, annotating species labels and, where possible, artifact conditions (blur, occlusion, infrared)",
        "Apply species-aware train/validation/test splits that avoid leakage between images from the same camera trap burst sequence",
        "Implement ResNet and Vision Transformer classifiers, both pretrained on large-scale image datasets and fine-tuned on the camera trap corpus",
        "Apply long-tailed recognition techniques such as class-balanced loss re-weighting or targeted re-sampling to address rare species underrepresentation",
        "Evaluate models on held-out data stratified by artifact type and, separately, on a held-out geographic region to test cross-site generalization",
        "Conduct error analysis on misclassified species pairs to identify visually confusable species groups requiring additional data or feature engineering",
    ],
    "unverified_suggestions": ["iNaturalist bird species dataset", "NABirds dataset", "Caltech Camera Traps dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds architecture comparison in a realistic setting (camera trap imagery) rather than curated benchmark photos, addressing a known gap in fine-grained classification research",
        "Explicit handling of long-tailed species distributions through class-balanced techniques reflects a practical understanding of real wildlife monitoring data",
        "Artifact-stratified evaluation (blur, occlusion, infrared) provides a more actionable diagnostic than a single aggregate accuracy number",
        "Cross-region generalization testing directly addresses a deployment-relevant question for conservation organizations operating in multiple sites",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) is not a dedicated wildlife imagery source, so the actual camera trap datasets remain unverified suggestions that may require ecological research data-sharing agreements.",
        "The proposal does not address how burst-sequence images from the same camera trigger event will be prevented from causing train/test leakage beyond a brief mention.",
        "There is limited discussion of how visually similar sibling species (which even expert ornithologists sometimes disagree on) will be handled in ground-truth labeling.",
    ],
    "missing_components": [
        "A detailed protocol for expert label verification or inter-annotator agreement on visually similar sibling species.",
        "A quantitative plan for detecting and preventing burst-sequence data leakage across train/validation/test splits.",
    ],
    "novelty_score": 0.60,
    "quality_score": 0.71,
}

CONFIDENCE_JSON = {"internal_consistency": 0.76, "specificity": 0.75, "completeness": 0.79}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning-based bird species identification from camera trap images, based on the retrieved abstract." for t in titles]
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
