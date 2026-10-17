import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Self-Supervised Representation Learning for Satellite-Based Crop Yield Estimation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a self-supervised pretraining framework for multi-temporal satellite imagery that produces transferable crop representations without requiring dense yield labels.",
        "Evaluate the downstream performance of self-supervised embeddings on county-level and field-level crop yield estimation across multiple crop types and growing seasons.",
        "Investigate the robustness of learned representations to cloud cover, sensor variability, and cross-region domain shift.",
        "Compare contrastive and masked-autoencoding self-supervised objectives for their suitability to multispectral time-series satellite data.",
    ],
    "research_questions": [
        "Can self-supervised representations learned from unlabeled Sentinel-2 and Landsat imagery match or exceed the accuracy of supervised CNN baselines for crop yield prediction?",
        "How does the choice of self-supervised pretraining objective (contrastive vs. masked image modeling) affect downstream yield estimation accuracy under limited-label conditions?",
        "To what extent do self-supervised embeddings generalize across geographically distinct agricultural regions and crop types without fine-tuning?",
        "What temporal sampling strategies best capture phenological growth stages relevant to yield formation in a self-supervised setting?",
    ],
    "keywords": [
        "self-supervised learning",
        "satellite remote sensing",
        "crop yield estimation",
        "contrastive representation learning",
        "multispectral time series",
    ],
    "research_scope": "This research focuses on applying self-supervised representation learning techniques to multispectral and multi-temporal satellite imagery, primarily from Sentinel-2 and Landsat missions, for the purpose of estimating crop yield at county and field scales. The scope includes pretraining on large unlabeled archives of agricultural remote sensing imagery, transferring learned representations to downstream yield regression tasks with limited labeled data, and evaluating generalization across staple crops such as maize, wheat, and soybean in different climatic regions. The work does not extend to real-time operational deployment or to non-optical sensing modalities such as SAR, though these are noted as promising extensions.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of contrastive self-supervised methods (e.g., SimCLR, MoCo variants) adapted to multispectral satellite bands rather than natural RGB imagery.",
        "Increasing use of masked autoencoders and vision transformers pretrained on large-scale unlabeled Earth observation archives such as SSL4EO-S12.",
        "Shift from single-image feature extraction toward temporal sequence modeling that captures crop phenology across the growing season.",
        "Combination of self-supervised pretraining with lightweight downstream regressors to reduce reliance on scarce ground-truth yield labels.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of contrastive versus generative (masked modeling) self-supervised objectives specifically for yield regression rather than land-cover classification.",
            "based_on_papers": ["Self-Supervised Learning in Remote Sensing: A Review"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness evaluation of self-supervised crop embeddings under realistic cloud occlusion and missing-observation patterns typical of operational satellite revisit schedules.",
            "based_on_papers": ["Deep Learning in Agriculture: A Survey"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-continental transfer studies that test whether representations pretrained on one agricultural region (e.g., US Corn Belt) retain predictive power for yield in structurally different smallholder farming systems.",
            "based_on_papers": ["Crop Yield Prediction Using Deep Neural Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design temporally-aware augmentation strategies (e.g., phenology-consistent cropping) for contrastive pretraining on satellite image time series.",
        "Investigate multimodal self-supervised fusion of optical, SAR, and weather data to improve yield estimation robustness under cloud cover.",
        "Develop benchmark datasets with dense, geographically diverse yield labels to standardize evaluation of self-supervised transfer performance.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "Copernicus Open Access Hub", "url": "https://scihub.copernicus.eu/", "description": "Satellite earth-observation data, commonly used for climate research.", "verified": True}
    ],
    "algorithms": [
        {"name": "Convolutional Neural Network (CNN)", "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network", "description": "A deep learning architecture widely used to extract spatial features from satellite imagery for downstream regression or classification tasks such as yield estimation.", "verified": True},
        {"name": "SimCLR (Simple Framework for Contrastive Learning of Visual Representations)", "url": "https://arxiv.org/abs/2002.05709", "description": "A self-supervised contrastive learning framework that learns image representations by maximizing agreement between augmented views of the same image, adaptable to multispectral satellite bands for pretraining crop yield models.", "verified": True},
    ],
    "evaluation_metrics": [
        "Root Mean Squared Error (RMSE) between predicted and observed crop yield",
        "Mean Absolute Error (MAE) at county and field level",
        "Coefficient of determination (R-squared) for yield regression fit",
        "Downstream linear-probe accuracy of frozen self-supervised embeddings",
    ],
    "implementation_steps": [
        "Assemble and preprocess a large unlabeled corpus of Sentinel-2 and Landsat multispectral image time series covering major crop-growing regions",
        "Apply cloud masking, atmospheric correction, and temporal compositing to produce analysis-ready image stacks",
        "Pretrain a self-supervised encoder (contrastive or masked-autoencoding) on the unlabeled imagery using spectral- and phenology-aware augmentations",
        "Freeze or fine-tune the pretrained encoder and attach a lightweight regression head trained on limited labeled county- or field-level yield data",
        "Evaluate downstream yield prediction performance against supervised CNN and traditional vegetation-index baselines",
        "Assess cross-region and cross-crop generalization by transferring the pretrained encoder to held-out geographic areas without further pretraining",
    ],
    "unverified_suggestions": [
        "Incorporate weather and soil covariates alongside satellite embeddings via late-fusion to potentially improve yield prediction accuracy.",
        "Explore vision transformer backbones as an alternative to CNNs for capturing longer-range spatial dependencies in field imagery.",
        "Use synthetic cloud-occlusion augmentation during pretraining to improve robustness to missing observations.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates the use of self-supervised learning to reduce dependence on scarce, costly-to-collect crop yield labels.",
        "The research questions are well-scoped and directly comparable to existing supervised baselines, enabling clear benchmarking.",
        "The methodology combines an established contrastive learning framework with a concrete, reproducible pipeline of preprocessing and evaluation steps.",
        "The plan appropriately separates pretraining data requirements (unlabeled, abundant) from fine-tuning data requirements (labeled, scarce), reflecting realistic data availability constraints.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small, so conclusions about which self-supervised objective performs best for yield regression specifically should be treated as preliminary rather than definitive.",
        "The proposal does not fully specify how ground-truth yield labels will be geographically aligned with satellite image patches, which is a common source of label noise in this domain.",
        "Cross-region generalization claims are not yet backed by pilot experiments, so the extent of domain shift robustness remains speculative.",
        "The interplay between temporal compositing choices and phenological stage coverage is under-specified and could materially affect downstream accuracy.",
    ],
    "missing_components": [
        "A concrete baseline comparison table specifying which supervised models the self-supervised approach will be benchmarked against.",
        "An explicit data licensing and access plan for combining Copernicus imagery with any proprietary or national agricultural yield statistics.",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to self-supervised satellite-based crop yield estimation, based on the retrieved abstract." for t in titles]
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
