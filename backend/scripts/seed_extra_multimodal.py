import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Multimodal Deep Learning for Sentiment Analysis in Social Media"

PLANNER_JSON = {
    "research_objectives": [
        "Design a multimodal deep learning architecture that jointly models text, images, and video frames from social media posts for sentiment classification",
        "Evaluate cross-modal fusion strategies (early, late, and attention-based fusion) for their impact on sentiment prediction accuracy",
        "Investigate the robustness of multimodal sentiment models to missing or noisy modalities common in real-world social media data",
        "Assess the interpretability of learned cross-modal attention weights in explaining sentiment predictions",
    ],
    "research_questions": [
        "How does fusing visual and textual modalities improve sentiment classification accuracy over unimodal text-only baselines on social media data?",
        "Which fusion architecture (early, late, or attention-based) best balances accuracy and computational cost for multimodal sentiment analysis?",
        "How resilient are multimodal sentiment models to missing modalities, such as posts with text but no accompanying image?",
        "Can cross-modal attention mechanisms provide human-interpretable explanations for sentiment predictions on social media posts?",
    ],
    "keywords": [
        "multimodal sentiment analysis",
        "cross-modal attention fusion",
        "social media affective computing",
        "vision-language transformers",
        "image-text sentiment classification",
    ],
    "research_scope": "This research focuses on supervised multimodal sentiment analysis for social media content that combines short-form text (tweets, captions) with a single accompanying image, using publicly available benchmark datasets such as MVSA and Twitter-based multimodal sentiment corpora. The scope excludes audio and long-form video modalities, real-time streaming inference, and multilingual sentiment transfer, concentrating instead on English-language, image-text pairs and their fusion architectures.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of transformer-based vision-language models (e.g., CLIP-style dual encoders) as backbones for multimodal sentiment fusion",
        "Increasing use of attention-based fusion mechanisms that dynamically weight modality contributions rather than fixed early or late fusion",
        "Rising interest in robustness to missing or corrupted modalities, reflecting the noisy and incomplete nature of real social media posts",
        "Emerging emphasis on interpretability and explainability of cross-modal attention to justify sentiment predictions to end users",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of multimodal sentiment models under adversarially perturbed images (e.g., memes with manipulated visual content) rather than naturally noisy data.",
            "based_on_papers": ["Multimodal Sentiment Analysis: A Survey of Methods, Trends and Challenges"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarking of fusion strategies across multiple social platforms (Twitter, Instagram, Reddit) to test cross-domain generalization of learned fusion weights.",
            "based_on_papers": ["Multimodal Sentiment Analysis on Social Media Using Deep Learning", "Attention-based Multimodal Fusion for Sentiment Classification"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative human-subject evaluation of whether cross-modal attention visualizations actually improve end-user trust or comprehension of sentiment predictions.",
            "based_on_papers": ["Interpretable Multimodal Learning for Affective Computing"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop benchmark suites with cross-platform and cross-domain splits to rigorously test the generalization of multimodal fusion architectures",
        "Explore lightweight, modality-dropout-robust fusion mechanisms suitable for on-device or low-latency social media moderation pipelines",
        "Combine attention-based interpretability methods with human evaluation studies to validate that explanations align with human sentiment judgments",
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
            "description": "Transformer-based image encoder that splits images into patches and processes them as token sequences, used here to extract visual embeddings from social media images for downstream sentiment fusion.",
            "verified": True,
        },
        {
            "name": "Multimodal Cross-Attention Fusion Network",
            "url": "https://arxiv.org/abs/1908.10084",
            "description": "A dual-encoder architecture with cross-modal attention layers that align text and image embeddings, allowing each modality to attend to the other before a joint sentiment classification head.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Macro F1-score",
        "Accuracy",
        "Area Under the ROC Curve (AUC)",
        "Cross-modal attention alignment score",
    ],
    "implementation_steps": [
        "Collect and preprocess paired text-image social media posts, normalizing text (tokenization, emoji handling) and resizing images to a fixed resolution",
        "Extract unimodal embeddings using a pretrained text encoder (e.g., BERT) and a pretrained image encoder (e.g., ViT)",
        "Implement and compare early fusion (concatenation before classification), late fusion (separate classifiers combined via weighted voting), and cross-attention fusion architectures",
        "Train each fusion variant on the labeled sentiment dataset using stratified train/validation/test splits with class-balanced sampling",
        "Evaluate models on held-out test data using macro F1, accuracy, and AUC, and conduct ablations with randomly dropped modalities to test robustness",
        "Visualize and qualitatively analyze cross-modal attention weights to assess interpretability of sentiment predictions",
    ],
    "unverified_suggestions": ["MVSA (Multi-View Sentiment Analysis) Dataset", "CrisisMMD", "Hugging Face Multimodal Hub"],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly articulated research objectives that connect multimodal fusion design choices directly to measurable sentiment classification outcomes",
        "Methodology proposes a well-motivated comparison across three distinct fusion strategies rather than committing prematurely to a single architecture",
        "Includes robustness testing under missing-modality conditions, which reflects realistic social media data quality issues",
        "Incorporates interpretability analysis of cross-modal attention, adding value beyond raw accuracy metrics",
    ],
    "weaknesses": [
        "The proposal does not specify how class imbalance in sentiment labels (commonly skewed toward neutral or positive) will be addressed beyond stratified sampling",
        "Evidence sufficiency is limited: the identified gaps rely on a small set of 8 retrieved papers, so claims about underexplored areas like adversarial robustness should be treated as preliminary rather than exhaustive",
        "Dataset fit caveats: the suggested unverified datasets (e.g., MVSA, CrisisMMD) have known licensing and image-availability restrictions that could affect reproducibility and were not validated against the whitelist",
        "No explicit baseline comparison against simpler unimodal text-only transformer models is described, making it harder to quantify the marginal benefit of multimodality",
    ],
    "missing_components": [
        "A concrete plan for hyperparameter tuning and computational budget across the three fusion architectures",
        "An error analysis section outlining how misclassified or ambiguous sentiment cases (e.g., sarcasm, mismatched text-image sentiment) will be examined",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to multimodal sentiment analysis on social media, based on the retrieved abstract." for t in titles]
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
