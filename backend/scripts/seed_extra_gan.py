import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Generative Adversarial Networks for Synthetic Medical Image Data Augmentation"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate the effectiveness of GAN-based synthetic image generation for augmenting limited medical imaging datasets across modalities such as MRI, CT, and histopathology",
        "Design GAN architectures and training strategies tailored to the low-data, high-resolution, and class-imbalanced characteristics typical of clinical imaging datasets",
        "Quantify the downstream impact of GAN-augmented training data on diagnostic classifier and segmentation model performance compared to traditional augmentation baselines",
        "Assess the clinical realism and diagnostic validity of GAN-synthesized medical images through both quantitative fidelity metrics and expert evaluation",
    ],
    "research_questions": [
        "Does augmenting training sets with GAN-synthesized medical images improve downstream diagnostic model accuracy relative to conventional geometric and intensity-based augmentation?",
        "Which GAN architectures (e.g., conditional GANs, StyleGAN variants, CycleGAN) best preserve clinically relevant pathological features when generating synthetic medical images?",
        "How does the amount and ratio of synthetic-to-real training data affect the generalization and calibration of downstream diagnostic classifiers?",
        "What risks of mode collapse, artifact generation, or diagnostic feature distortion arise when applying GANs to small, class-imbalanced medical imaging datasets?",
    ],
    "keywords": [
        "generative adversarial network",
        "medical image synthesis",
        "data augmentation",
        "synthetic medical imaging",
        "conditional image generation",
    ],
    "research_scope": (
        "This research focuses on GAN-based generative models used to synthesize medical images for the purpose of "
        "augmenting training data in downstream diagnostic classification and segmentation tasks. It covers "
        "conditional and unconditional GAN architectures, image-to-image translation approaches (e.g., CycleGAN), "
        "and evaluation of synthetic image fidelity and downstream task performance across imaging modalities "
        "including radiography, MRI, CT, and histopathology. The scope excludes diffusion-model-based generation, "
        "3D volumetric synthesis pipelines, federated or privacy-preserving generation techniques, and clinical "
        "deployment or regulatory approval considerations, which are treated as out of scope for this analysis."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of conditional GANs to generate class-specific pathological images (e.g., tumor-present vs. tumor-absent) to address class imbalance in medical datasets",
        "Adoption of image-to-image translation frameworks such as CycleGAN for cross-modality synthesis (e.g., MRI-to-CT) when paired training data is unavailable",
        "Growing reliance on hybrid evaluation protocols combining quantitative fidelity metrics (FID, IS) with radiologist-in-the-loop visual Turing tests to validate synthetic image realism",
        "Emergence of progressive and style-based GAN architectures (e.g., StyleGAN variants) for generating high-resolution synthetic histopathology and dermoscopy images",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of how varying the synthetic-to-real training data ratio affects downstream diagnostic classifier calibration, rather than accuracy alone, across multiple imaging modalities.",
            "based_on_papers": ["Generative Adversarial Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative characterization of failure modes where GAN-generated medical images introduce spurious pathological artifacts that mislead downstream classifiers despite passing standard fidelity metrics like FID.",
            "based_on_papers": ["Image-to-Image Translation with Conditional Adversarial Networks", "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized, cross-institution protocols for radiologist-in-the-loop evaluation of synthetic image diagnostic validity that go beyond single-site visual Turing tests.",
            "based_on_papers": ["Generative Adversarial Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop calibration-aware evaluation frameworks that jointly report classifier accuracy and confidence reliability under varying synthetic data ratios",
        "Investigate anomaly-detection safeguards that flag GAN-generated images containing implausible or spurious pathological artifacts before they enter training pipelines",
        "Establish multi-institution, radiologist-adjudicated benchmarks for assessing the diagnostic realism of synthetic medical images across modalities",
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
            "name": "Generative Adversarial Network (GAN)",
            "url": "https://arxiv.org/abs/1406.2661",
            "description": "Foundational adversarial framework pairing a generator and discriminator in a minimax game, serving as the base architecture for synthetic medical image generation experiments.",
            "verified": True,
        },
        {
            "name": "Conditional GAN (pix2pix)",
            "url": "https://arxiv.org/abs/1611.07004",
            "description": "Conditional image-to-image translation architecture that generates class- or condition-specific synthetic images, used to produce pathology-labeled synthetic samples for targeted class-imbalance augmentation.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Frechet Inception Distance (FID)",
        "Inception Score (IS)",
        "Downstream classifier accuracy and AUC with synthetic augmentation",
        "Expected calibration error (ECE) of downstream classifiers",
    ],
    "implementation_steps": [
        "Curate and preprocess real medical imaging datasets across target modalities, applying consistent normalization and class labeling",
        "Select and implement baseline and conditional GAN architectures suited to the resolution and class distribution of each modality",
        "Train GAN models with regularization and stabilization techniques to mitigate mode collapse on small clinical datasets",
        "Generate synthetic image sets at varying synthetic-to-real ratios and filter outputs using automated artifact and fidelity checks",
        "Train downstream diagnostic classifiers and segmentation models on real, synthetic-augmented, and traditionally-augmented datasets for comparison",
        "Evaluate synthetic image fidelity (FID, IS) and downstream task performance (accuracy, AUC, calibration) and conduct expert visual review of a sample of generated images",
    ],
    "unverified_suggestions": ["CheXpert", "BraTS", "ISIC Archive"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly frames a testable comparison between GAN-augmented and conventionally-augmented training data for downstream diagnostic performance",
        "The methodology includes a concrete, staged pipeline from GAN training through synthetic filtering to downstream evaluation, which supports reproducibility",
        "Evaluation metrics appropriately combine generative fidelity measures (FID, IS) with downstream task metrics and calibration, avoiding reliance on fidelity metrics alone",
        "The identified gaps are specific and directly extend limitations of the cited conditional and image-to-image translation GAN literature",
    ],
    "weaknesses": [
        "The proposal does not fully specify thresholds or criteria for the automated artifact-filtering step, leaving synthetic data quality control partly underdefined",
        "Evidence sufficiency is limited: only a small set of retrieved papers directly evaluate downstream diagnostic impact of GAN augmentation, so conclusions should be treated as preliminary pending broader validation",
        "Dataset fit caveats apply since the whitelisted repository (OpenML) may not directly host the specialized clinical imaging benchmarks implied by the unverified suggestions such as CheXpert and BraTS",
        "The plan does not address potential patient privacy or re-identification risks that can arise when GANs memorize and reproduce features of training images",
    ],
    "missing_components": [
        "An explicit plan for statistical significance testing when comparing downstream classifier performance across augmentation strategies",
        "A defined protocol for radiologist-in-the-loop review, including reviewer qualifications and inter-rater agreement measurement",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.82, "specificity": 0.79, "completeness": 0.84}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to GAN-based synthetic medical image augmentation, based on the retrieved abstract." for t in titles]
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
