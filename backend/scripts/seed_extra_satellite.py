import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Contrastive Self-Supervised Pretraining for Satellite Image Change Detection"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a contrastive self-supervised pretraining framework for satellite imagery that learns change-sensitive representations without requiring pixel-level change labels",
        "Evaluate whether pretraining on large unlabeled multi-temporal satellite archives improves downstream change detection accuracy compared to supervised-from-scratch and ImageNet-pretrained baselines",
        "Investigate the robustness of learned representations to seasonal variation, illumination shifts, and sensor noise across temporal image pairs",
        "Assess the label efficiency of the pretrained representations when fine-tuned on small annotated change detection datasets",
    ],
    "research_questions": [
        "Can contrastive self-supervised objectives applied to unlabeled bi-temporal or multi-temporal satellite image pairs learn representations that transfer effectively to change detection?",
        "How do augmentation strategies designed for natural images (cropping, color jitter) need to be adapted for multi-spectral and multi-temporal remote sensing data?",
        "What is the minimum amount of labeled change detection data needed to reach competitive performance when starting from a contrastively pretrained encoder?",
        "How does contrastive pretraining affect false-positive change detections caused by seasonal vegetation change or cloud/shadow artifacts?",
    ],
    "keywords": [
        "self-supervised learning",
        "contrastive representation learning",
        "satellite image change detection",
        "remote sensing pretraining",
        "bi-temporal image analysis",
    ],
    "research_scope": "This research focuses on designing and evaluating contrastive self-supervised pretraining methods (e.g., SimCLR- and MoCo-style frameworks) adapted to multi-spectral, multi-temporal satellite imagery, with the goal of producing encoders that transfer well to downstream binary and semantic change detection tasks. The scope covers pretraining data curation from open earth-observation archives, augmentation design tailored to remote sensing characteristics (seasonal variation, radiometric differences, cloud cover), and empirical evaluation on standard change detection benchmarks under both low-label and full-label fine-tuning regimes. It does not cover real-time streaming change detection or non-optical sensing modalities such as SAR, though these are noted as future extensions.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of Siamese and transformer-based architectures (e.g., ChangeFormer, BIT) for supervised bi-temporal change detection",
        "Increasing use of self-supervised contrastive pretraining (SimCLR, MoCo, BYOL) in general computer vision, with early adaptations appearing for remote sensing scene classification",
        "Expanding availability of large open multi-temporal satellite archives (Sentinel-2, Landsat) enabling large-scale unlabeled pretraining corpora",
        "Shift toward foundation-model style pretraining for earth observation, though most efforts target classification and segmentation rather than change detection specifically",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic study of how contrastive pretraining objectives should be modified to explicitly encode temporal change (as opposed to invariance to it) rather than reusing invariance-based objectives designed for single-image classification.",
            "based_on_papers": ["A Simple Framework for Contrastive Learning of Visual Representations"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: limited evaluation of contrastively pretrained encoders under domain shift between the unlabeled pretraining archive and the labeled fine-tuning dataset (e.g., different sensors, resolutions, or geographic regions).",
            "based_on_papers": ["Momentum Contrast for Unsupervised Visual Representation Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of how much labeled change detection data is actually saved by self-supervised pretraining, i.e., label-efficiency curves comparing pretrained versus randomly initialized encoders across varying annotation budgets.",
            "based_on_papers": ["Fully Convolutional Siamese Networks for Change Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design change-aware contrastive objectives that treat co-registered bi-temporal patches as positive pairs only when no semantic change has occurred, using weak heuristics or auxiliary signals to filter true negatives",
        "Build cross-sensor and cross-region benchmarks to systematically evaluate the transferability of self-supervised remote sensing encoders",
        "Study curriculum or hybrid pretraining strategies that combine contrastive objectives with change-specific pretext tasks (e.g., predicting temporal difference masks)",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "Copernicus Open Access Hub",
            "url": "https://scihub.copernicus.eu/",
            "description": "Satellite earth-observation data, commonly used for climate research.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "description": "Backbone architecture (e.g., ResNet-50) used as the encoder for both contrastive pretraining and the downstream change detection head.",
            "verified": True,
        },
        {
            "name": "SimCLR (Contrastive Learning Framework)",
            "url": "https://arxiv.org/abs/2002.05709",
            "description": "Self-supervised contrastive learning framework that maximizes agreement between differently augmented views of the same image via a normalized temperature-scaled cross-entropy loss, adapted here to bi-temporal satellite patches.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision",
        "Recall",
        "F1-score",
        "Intersection over Union (IoU)",
    ],
    "implementation_steps": [
        "Curate an unlabeled pretraining corpus by downloading co-registered multi-temporal Sentinel-2 tiles from the Copernicus Open Access Hub and applying cloud/shadow filtering",
        "Define remote-sensing-appropriate augmentations (radiometric jitter, random multi-spectral band dropout, rotation, and temporal pair sampling) for the contrastive pretext task",
        "Pretrain a ResNet-50 encoder using a SimCLR-style contrastive loss on the unlabeled bi-temporal patches",
        "Attach a lightweight Siamese change detection head to the pretrained encoder and fine-tune on a labeled change detection dataset such as LEVIR-CD or OSCD",
        "Evaluate downstream performance under multiple label budgets (e.g., 1%, 10%, 100% of labeled data) to quantify label efficiency gains from pretraining",
        "Compare against ImageNet-pretrained and randomly initialized baselines using precision, recall, F1-score, and IoU on held-out change detection test regions",
    ],
    "unverified_suggestions": [
        "Consider incorporating SAR imagery (e.g., Sentinel-1) as an auxiliary modality to improve robustness to cloud cover during pretraining",
        "Explore vision transformer backbones (e.g., Swin Transformer) as an alternative to CNN encoders for capturing longer-range spatial context in change regions",
        "Investigate self-supervised change-mask pseudo-labeling as an additional pretext task to complement contrastive pretraining",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates the use of self-supervised pretraining to reduce reliance on scarce, expensive pixel-level change annotations",
        "Augmentation design is thoughtfully adapted to remote sensing characteristics rather than naively reusing natural-image augmentations",
        "The evaluation plan includes label-efficiency curves, which directly test the core hypothesis rather than only reporting a single fine-tuned accuracy number",
        "Grounding the methodology in established frameworks (SimCLR, Siamese CNNs) and public benchmarks (LEVIR-CD, OSCD) improves reproducibility",
    ],
    "weaknesses": [
        "The evidence base is limited to 8 retrieved papers, which may not sufficiently cover recent transformer-based change detection or foundation-model pretraining literature",
        "The proposal does not specify how positive/negative pairs will be constructed for bi-temporal contrastive learning, which is a nontrivial design choice distinct from standard single-image SimCLR",
        "No baseline comparison against existing self-supervised remote sensing pretraining methods (e.g., SeCo, GASSL) is planned, only against ImageNet and random initialization",
        "Potential mismatch between the unlabeled pretraining source (Sentinel-2, 10m resolution) and downstream benchmark datasets that may use different sensors or resolutions is not addressed",
    ],
    "missing_components": [
        "A explicit ablation plan isolating the contribution of temporal pair construction versus generic contrastive pretraining",
        "A discussion of compute budget and pretraining scale needed to reach the claimed representation quality",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to contrastive self-supervised pretraining for satellite change detection, based on the retrieved abstract." for t in titles]
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
