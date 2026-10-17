import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Vision Transformers for Few-Shot Medical Image Classification"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate how well Vision Transformer (ViT) architectures generalize to medical imaging tasks when only a small number of labeled examples per class are available",
        "Design and benchmark few-shot learning strategies (e.g., meta-learning, prototypical networks, prompt-based adaptation) built on top of pretrained ViT backbones for medical image classification",
        "Quantify the impact of self-supervised and transfer pretraining on ViT few-shot performance across heterogeneous medical imaging modalities (radiology, histopathology, dermoscopy)",
        "Identify architectural or training modifications that mitigate ViT's data-hunger relative to CNNs in low-data medical imaging regimes",
    ],
    "research_questions": [
        "Can Vision Transformers match or exceed CNN-based few-shot classifiers on medical imaging benchmarks when pretraining and adaptation strategies are held constant?",
        "Which few-shot adaptation techniques (fine-tuning, prototypical heads, adapters, prompt tuning) yield the best accuracy-to-data-efficiency tradeoff for ViT on medical images?",
        "How does the choice of pretraining corpus (natural images vs. domain-specific medical images) affect few-shot transfer performance of ViTs?",
        "What is the sensitivity of ViT few-shot performance to class imbalance and modality shift across different medical imaging datasets?",
    ],
    "keywords": [
        "vision transformer",
        "few-shot learning",
        "medical image classification",
        "self-supervised pretraining",
        "domain transfer medical imaging",
    ],
    "research_scope": (
        "This research focuses on supervised and self-supervised Vision Transformer architectures applied to "
        "few-shot medical image classification tasks, spanning modalities such as radiography, histopathology, "
        "and dermoscopy. It covers meta-learning and parameter-efficient adaptation strategies (prototypical "
        "networks, adapters, prompt tuning, linear probing) applied on top of pretrained ViT backbones, and "
        "compares them against CNN and hybrid CNN-Transformer baselines. The scope excludes purely generative "
        "or segmentation-focused approaches, 3D volumetric imaging pipelines, and clinical deployment or "
        "regulatory considerations, which are treated as out of scope for this analysis."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of self-supervised pretraining (e.g., masked autoencoders, DINO) on large natural-image corpora before fine-tuning ViTs on small medical datasets",
        "Adoption of parameter-efficient adaptation methods such as adapters and prompt tuning to reduce overfitting when fine-tuning ViTs with few labeled medical examples",
        "Increasing interest in hybrid CNN-Transformer architectures that combine convolutional inductive biases with transformer global attention for low-data regimes",
        "Emergence of domain-specific foundation models pretrained directly on medical imaging archives to reduce the natural-to-medical domain gap",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of few-shot adaptation strategies (prototypical heads vs. adapters vs. prompt tuning) applied to the same ViT backbone across multiple medical imaging modalities under a unified evaluation protocol.",
            "based_on_papers": ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of how the degree of domain shift between pretraining corpus (natural images) and target medical modality affects few-shot sample efficiency.",
            "based_on_papers": ["Emerging Properties in Self-Supervised Vision Transformers", "Masked Autoencoders Are Scalable Vision Learners"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness evaluation of few-shot ViT classifiers under realistic class imbalance and label noise conditions typical of clinical datasets.",
            "based_on_papers": ["Masked Autoencoders Are Scalable Vision Learners"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop unified benchmarks that evaluate few-shot ViT adaptation strategies consistently across radiology, histopathology, and dermoscopy datasets",
        "Investigate lightweight domain-adaptive pretraining objectives that narrow the natural-to-medical image gap before few-shot fine-tuning",
        "Explore uncertainty-aware few-shot classifiers that flag low-confidence predictions in imbalanced or noisy clinical label settings",
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
            "name": "Vision Transformer (ViT)",
            "url": "https://arxiv.org/abs/2010.11929",
            "description": "Transformer architecture that treats image patches as token sequences and applies self-attention for global context modeling, serving as the primary backbone for few-shot adaptation experiments.",
            "verified": True,
        },
        {
            "name": "Prototypical Networks",
            "url": "https://arxiv.org/abs/1703.05175",
            "description": "Metric-based few-shot learning baseline that classifies query examples by distance to class prototypes computed from support embeddings, used as a comparison adaptation head on top of ViT features.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Few-shot classification accuracy (k-shot, n-way)",
        "Area under the ROC curve (AUC)",
        "Macro-averaged F1 score",
        "Expected calibration error (ECE)",
    ],
    "implementation_steps": [
        "Curate and preprocess medical imaging datasets across target modalities, ensuring consistent patch-compatible resolution and normalization",
        "Pretrain or select a pretrained ViT backbone using self-supervised objectives on natural and, where available, domain-specific medical corpora",
        "Construct few-shot episodes (n-way, k-shot) with stratified sampling to control class imbalance in support and query sets",
        "Implement and train adaptation heads (prototypical network, adapter modules, prompt tuning) on top of the frozen or partially fine-tuned ViT backbone",
        "Evaluate each adaptation strategy across modalities using held-out few-shot episodes and compute accuracy, AUC, F1, and calibration metrics",
        "Perform ablations varying shot count, pretraining corpus, and degree of backbone fine-tuning to isolate sources of performance variation",
    ],
    "unverified_suggestions": ["MedMNIST", "CheXpert", "ISIC Archive"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly articulates a focused research question comparing ViT and CNN few-shot performance under controlled adaptation strategies",
        "The methodology specifies a concrete, reproducible experimental pipeline spanning pretraining, episodic sampling, and multiple adaptation heads",
        "Evaluation metrics are appropriate for imbalanced medical classification tasks, including calibration error alongside accuracy and AUC",
        "The identified gaps are specific and build logically on limitations of the cited self-supervised ViT literature",
    ],
    "weaknesses": [
        "The proposal does not fully specify how domain shift severity will be quantified across modalities, leaving the comparison partly qualitative",
        "Evidence sufficiency is limited: only a small set of retrieved papers directly benchmark few-shot ViT adaptation on medical images, so conclusions should be treated as preliminary until validated on the proposed unified benchmark",
        "Dataset fit caveats apply since the whitelisted dataset repository (OpenML) may not directly host the specialized medical imaging benchmarks implied by the unverified suggestions",
        "The plan does not address computational cost tradeoffs between full fine-tuning and parameter-efficient adaptation methods",
    ],
    "missing_components": [
        "A clear baseline comparison protocol against non-transformer few-shot methods beyond prototypical networks",
        "An explicit plan for statistical significance testing across few-shot episodes and modalities",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to few-shot vision transformer adaptation for medical imaging, based on the retrieved abstract." for t in titles]
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
