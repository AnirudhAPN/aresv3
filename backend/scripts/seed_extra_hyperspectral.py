import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Hyperspectral Image Classification Using 3D Convolutional Networks for Mineral Exploration"

PLANNER_JSON = {
    "research_objectives": [
        "Investigate how 3D convolutional neural network architectures can jointly exploit spectral and spatial correlations in hyperspectral imagery to improve classification of mineral and lithological classes compared to 2D CNN and traditional spectral-only classifiers.",
        "Characterize the impact of dimensionality-reduction and band-selection strategies (e.g., PCA, autoencoders) on 3D-CNN classification accuracy and computational cost for large-scale hyperspectral mineral exploration datasets.",
        "Design and evaluate training strategies that address the scarcity of labeled ground-truth mineral samples, including transfer learning and data augmentation, for 3D-CNN-based hyperspectral classification.",
        "Assess the generalization of 3D-CNN mineral classifiers across different sensors (e.g., AVIRIS, Hyperion, PRISMA) and geographic terrains typical of mineral exploration sites.",
    ],
    "research_questions": [
        "How does incorporating 3D spectral-spatial convolutions instead of purely spectral (1D) or 2D spatial-only convolutions affect classification accuracy for distinguishing mineral classes with subtle spectral absorption signatures?",
        "What band-selection or dimensionality-reduction preprocessing best balances classification accuracy against the computational and memory cost of training 3D CNNs on high-band-count hyperspectral cubes?",
        "How much labeled training data is required for 3D-CNN mineral classifiers to outperform classical spectral classifiers (e.g., Spectral Angle Mapper, Support Vector Machines), and can transfer learning reduce this requirement?",
        "How robust are 3D-CNN hyperspectral mineral classifiers to cross-sensor and cross-site domain shift compared to shallower baseline models?",
    ],
    "keywords": [
        "hyperspectral image classification",
        "3D convolutional neural networks",
        "mineral exploration remote sensing",
        "spectral-spatial feature learning",
        "band selection dimensionality reduction",
    ],
    "research_scope": (
        "This research focuses on 3D convolutional neural network approaches to classifying minerals and "
        "lithological units from hyperspectral remote sensing imagery, covering spectral-spatial feature "
        "extraction, dimensionality-reduction preprocessing (PCA, autoencoders), and training under limited "
        "labeled-sample conditions typical of exploration geology. It includes evaluation against classical "
        "spectral classifiers (Spectral Angle Mapper, Support Vector Machines) and 2D-CNN baselines, and "
        "considers generalization across airborne and spaceborne hyperspectral sensors (AVIRIS, Hyperion, "
        "PRISMA) and different geological terrains. It excludes non-imaging spectroscopy methods, multispectral "
        "(as opposed to hyperspectral) classification pipelines, and downstream economic or geostatistical "
        "ore-grade estimation that falls outside image classification itself."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of 3D-CNN and hybrid 2D/3D-CNN architectures for hyperspectral land-cover and mineral classification, motivated by their ability to jointly learn spectral and spatial features.",
        "Increasing use of dimensionality-reduction front-ends (PCA, factor analysis, autoencoders) to compress hundreds of spectral bands before 3D convolution, reducing the curse of dimensionality and training cost.",
        "Rising interest in attention mechanisms and hybrid CNN-transformer models layered on top of 3D convolutional backbones to better capture long-range spectral dependencies relevant to mineral absorption features.",
        "Persistent reliance on a small set of well-studied benchmark hyperspectral scenes (e.g., Indian Pines, Pavia University, Salinas) that are agricultural or urban rather than mineral-exploration-specific, limiting demonstrated applicability to geological mapping.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic benchmarking of 3D-CNN architectures specifically on mineral and lithological mapping tasks using exploration-grade hyperspectral data (e.g., AVIRIS or PRISMA scenes over mining districts) rather than the standard agricultural or urban benchmark scenes.",
            "based_on_papers": ["3-D Deep Learning Approach for Remote Sensing Image Classification"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantification of how few labeled mineral ground-truth samples, typical of field-survey-limited exploration settings, 3D-CNN models need before their accuracy advantage over classical spectral classifiers like SAM or SVM disappears.",
            "based_on_papers": ["Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-sensor and cross-site generalization of 3D-CNN mineral classifiers, i.e., whether a model trained on one hyperspectral sensor and geology generalizes to a different sensor or terrain without retraining.",
            "based_on_papers": ["Hyperspectral Image Classification With Deep Learning Models"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Curating and releasing labeled hyperspectral mineral-exploration benchmark datasets spanning multiple sensors and geological settings to support standardized 3D-CNN evaluation.",
        "Developing few-shot or self-supervised pretraining strategies tailored to mineral spectral libraries (e.g., the USGS spectral library) to reduce dependence on scarce field-labeled data.",
        "Investigating lightweight or factorized 3D-convolution variants (e.g., separable 3D convolutions) to make hyperspectral mineral classification feasible for onboard or real-time exploration platforms.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "Copernicus Open Access Hub", "url": "https://scihub.copernicus.eu/", "description": "Satellite earth-observation data, commonly used for climate research.", "verified": True}],
    "algorithms": [
        {
            "name": "3D Convolutional Neural Network (3D-CNN)",
            "url": "https://arxiv.org/abs/1610.00960",
            "description": "Convolutional architecture that applies 3D kernels jointly across the spatial and spectral dimensions of a hyperspectral cube, allowing the network to learn joint spectral-spatial discriminative features for mineral classification.",
            "verified": True,
        },
        {
            "name": "Principal Component Analysis (PCA)",
            "url": "https://en.wikipedia.org/wiki/Principal_component_analysis",
            "description": "Linear dimensionality-reduction technique commonly applied to hyperspectral cubes before 3D-CNN input to compress redundant spectral bands and reduce computational cost while retaining most spectral variance.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Overall Accuracy (OA)",
        "Average Accuracy (AA) across mineral classes",
        "Kappa coefficient",
        "Per-class F1-score for minority mineral classes",
    ],
    "implementation_steps": [
        "Acquire and radiometrically and atmospherically correct hyperspectral cubes covering the target mineral exploration site.",
        "Apply dimensionality reduction (e.g., PCA) to the spectral dimension to reduce band count while preserving discriminative spectral variance.",
        "Extract 3D spatial-spectral patches centered on labeled pixels to form training, validation, and test samples.",
        "Train the 3D-CNN classifier on labeled patches, using data augmentation (rotation, flipping) to mitigate limited mineral ground-truth availability.",
        "Tune hyperparameters and monitor for overfitting given the typically small number of labeled mineral samples relative to spectral dimensionality.",
        "Evaluate the trained model on held-out spatial regions and, where possible, on an independent sensor or site to assess generalization.",
    ],
    "unverified_suggestions": ["AVIRIS airborne hyperspectral mineral mapping scenes", "USGS Spectral Library for mineral reference spectra", "PRISMA satellite hyperspectral mission data"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal correctly identifies 3D convolution as a principled way to jointly exploit spectral and spatial structure in hyperspectral cubes, rather than treating bands as independent channels.",
        "Including PCA-based dimensionality reduction as a preprocessing step reflects an accurate and practical understanding of the computational challenges posed by high band-count hyperspectral data.",
        "The evaluation metrics (OA, AA, Kappa, per-class F1) are well matched to the class-imbalance issues typical of mineral classification, where economically important minerals are often minority classes.",
        "The implementation plan appropriately foregrounds data augmentation and careful validation, acknowledging the limited-labeled-data regime typical of mineral exploration surveys.",
    ],
    "weaknesses": [
        "Copernicus Open Access Hub, the only verified dataset in the methodology, provides Sentinel multispectral rather than hyperspectral imagery and is oriented toward climate and land monitoring rather than the airborne or spaceborne hyperspectral sensors (e.g., AVIRIS, PRISMA) needed for mineral exploration, so dataset fit should be revisited.",
        "The evidence base draws on a limited set of retrieved papers, most benchmarked on agricultural or urban hyperspectral scenes rather than mineral exploration data, so claims about applicability to geological mapping remain preliminary.",
        "The proposal does not specify how atmospheric correction and sensor-specific radiometric calibration differences will be handled, which can substantially affect spectral signatures used for mineral discrimination.",
        "There is limited discussion of the computational cost and memory footprint of 3D convolutions at full hyperspectral band counts, which could be a practical barrier for large exploration-area coverage.",
    ],
    "missing_components": [
        "A concrete plan for obtaining or constructing labeled mineral ground-truth data (e.g., via field spectroscopy or existing geological survey maps) to train and validate the classifier.",
        "An ablation comparing 3D-CNN performance against simpler 2D-CNN and classical spectral classifiers (SAM, SVM) to isolate the benefit attributable specifically to 3D spectral-spatial convolution.",
    ],
    "novelty_score": 0.71,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to 3D-CNN-based hyperspectral mineral classification, based on the retrieved abstract." for t in titles]
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
