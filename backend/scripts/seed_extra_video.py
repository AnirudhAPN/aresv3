import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Self-Supervised Contrastive Learning for Video Action Recognition"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a self-supervised contrastive learning framework that learns transferable video representations without manual action labels.",
        "Design temporally-aware augmentation and sampling strategies that produce informative positive and negative pairs from unlabeled video clips.",
        "Evaluate the learned representations on downstream action recognition benchmarks via linear probing and fine-tuning.",
        "Analyze the impact of spatial versus temporal augmentation choices on the quality of learned video representations.",
    ],
    "research_questions": [
        "Can contrastive pretraining on unlabeled video data match or exceed supervised pretraining for downstream action recognition accuracy?",
        "Which temporal augmentation strategies (e.g., clip sampling, frame dropout, speed perturbation) most improve the discriminability of learned video embeddings?",
        "How does the choice of negative sampling strategy affect convergence and downstream transfer performance in video contrastive learning?",
        "Does combining spatial and temporal contrastive objectives improve robustness to viewpoint and motion variation compared to single-objective approaches?",
    ],
    "keywords": [
        "self-supervised video representation learning",
        "contrastive learning action recognition",
        "video temporal augmentation",
        "instance discrimination video",
        "spatiotemporal representation learning",
    ],
    "research_scope": (
        "This research focuses on self-supervised contrastive learning methods applied to RGB video data for the "
        "downstream task of human action recognition, covering pretraining objectives (e.g., instance discrimination, "
        "momentum contrast, clustering-based approaches), temporal and spatial augmentation strategies, and evaluation "
        "protocols such as linear probing and fine-tuning on standard trimmed-video benchmarks. It excludes "
        "multimodal approaches that rely on audio, text, or optical flow as auxiliary supervision, excludes "
        "fully-supervised action recognition baselines except as reference points, and excludes online/streaming "
        "action detection or untrimmed video localization, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of momentum-based contrastive frameworks (e.g., MoCo-style memory queues) adapted from image self-supervision to spatiotemporal video encoders.",
        "Increasing use of temporally-aware augmentation pipelines, such as variable-speed clip sampling and frame-order shuffling, to construct positive pairs that encode motion information.",
        "Shift toward transformer-based video backbones combined with contrastive objectives as an alternative to 3D convolutional encoders.",
        "Rising interest in combining contrastive pretraining with lightweight fine-tuning or linear probing to reduce the labeled-data requirements of downstream action recognition.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of negative sampling strategies (hard negative mining vs. random in-batch negatives) specifically for video clips with overlapping temporal content.",
            "based_on_papers": ["A Simple Framework for Contrastive Learning of Visual Representations", "Momentum Contrast for Unsupervised Visual Representation Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of contrastive video representations under domain shift, such as transfer from third-person to egocentric viewpoints.",
            "based_on_papers": ["Video Representation Learning by Dense Predictive Coding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: computational cost and energy efficiency comparisons between contrastive video pretraining approaches and their supervised counterparts at equivalent downstream accuracy.",
            "based_on_papers": ["A Simple Framework for Contrastive Learning of Visual Representations", "Video Representation Learning by Dense Predictive Coding"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Investigate curriculum-based negative mining that progressively introduces harder temporal negatives as training converges.",
        "Explore unified spatiotemporal contrastive objectives that jointly optimize for appearance invariance and motion sensitivity.",
        "Study label-efficient adaptation protocols that combine contrastive pretraining with few-shot fine-tuning for low-resource action recognition datasets.",
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
            "name": "3D Convolutional Neural Network (3D-CNN / C3D)",
            "url": "https://arxiv.org/abs/1412.0767",
            "description": "A spatiotemporal convolutional architecture that extracts joint appearance and motion features directly from stacked video frames, commonly used as a backbone encoder for video contrastive pretraining.",
            "verified": True,
        },
        {
            "name": "SimCLR (Simple Framework for Contrastive Learning)",
            "url": "https://arxiv.org/abs/2002.05709",
            "description": "A contrastive self-supervised learning framework that maximizes agreement between differently augmented views of the same instance via a normalized temperature-scaled cross-entropy loss, adaptable to video by treating temporally augmented clips as positive pairs.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Top-1 accuracy on linear probing",
        "Top-5 accuracy on fine-tuned action recognition",
        "Mean average precision (mAP) for multi-label action clips",
        "Downstream transfer accuracy under limited-label fine-tuning",
    ],
    "implementation_steps": [
        "Curate and preprocess unlabeled video clips, extracting fixed-length frame sequences at a consistent sampling rate",
        "Implement temporal and spatial augmentation pipelines, including random clip sampling, speed perturbation, and spatial cropping",
        "Build a spatiotemporal encoder backbone paired with a projection head for contrastive embedding",
        "Pretrain the encoder using a contrastive loss with positive pairs from augmented views of the same clip and negatives from other clips in the batch or memory queue",
        "Evaluate learned representations via linear probing on a labeled action recognition benchmark by freezing the encoder and training a linear classifier",
        "Fine-tune the full network end-to-end on the downstream labeled dataset and compare accuracy against supervised-pretraining baselines",
    ],
    "unverified_suggestions": ["Kinetics-400", "UCF101", "Something-Something V2"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly grounds its objectives in established contrastive learning frameworks with concrete architectural references.",
        "The implementation steps form a coherent, reproducible pipeline from data preprocessing through downstream evaluation.",
        "The research questions are specific and testable, focusing on concrete design choices like augmentation and negative sampling.",
        "The scope explicitly excludes multimodal and streaming variants, keeping the research focus tractable and well-bounded.",
    ],
    "weaknesses": [
        "The proposal does not specify sample sizes or compute budgets, making it hard to assess evidence sufficiency for the planned experiments.",
        "Dataset fit is uncertain: the whitelisted OpenML repository is not a natural source for large-scale video action datasets, so the proposed unverified suggestions (Kinetics-400, UCF101) would need independent verification before use.",
        "No ablation plan is specified for isolating the individual contribution of spatial versus temporal augmentations.",
        "The proposal lacks discussion of statistical significance testing or confidence intervals for comparing downstream accuracy across methods.",
    ],
    "missing_components": [
        "A discussion of compute and data requirements needed to reproduce the contrastive pretraining stage at scale.",
        "An explicit baseline comparison protocol against fully supervised action recognition models trained on the same backbone.",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to self-supervised contrastive video representation learning, based on the retrieved abstract." for t in titles]
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
