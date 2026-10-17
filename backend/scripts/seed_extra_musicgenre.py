import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Attention Mechanisms for Music Genre Classification and Recommendation"

PLANNER_JSON = {
    "research_objectives": [
        "Investigate how self-attention mechanisms can capture long-range temporal and spectral dependencies in audio signals to improve music genre classification accuracy over CNN- and RNN-only baselines.",
        "Evaluate the effectiveness of attention-based architectures (e.g., Transformer encoders applied to spectrogram or embedding sequences) for distinguishing genres with overlapping timbral and rhythmic characteristics.",
        "Assess how attention weights learned for genre classification can be repurposed or jointly trained to support content-based music recommendation, linking classification confidence to recommendation relevance.",
        "Analyze the interpretability benefits of attention maps for identifying which time-frequency regions of a track most influence genre and recommendation decisions.",
    ],
    "research_questions": [
        "How do self-attention and Transformer-based architectures compare to CNN and RNN baselines in classifying music genres from mel-spectrogram or raw waveform inputs?",
        "Can cross-attention between audio embeddings and metadata (artist, tempo, lyrics) improve genre classification accuracy for ambiguous or fusion-genre tracks?",
        "How can attention weights derived from a genre classifier be leveraged as a similarity signal for content-based music recommendation?",
        "What is the trade-off between model complexity/inference latency and classification accuracy when scaling attention-based models for real-time recommendation pipelines?",
    ],
    "keywords": [
        "attention mechanisms",
        "music genre classification",
        "music recommendation systems",
        "transformer audio models",
        "mel-spectrogram representation learning",
    ],
    "research_scope": "This research focuses on attention-based deep learning architectures — including self-attention, Transformer encoders, and cross-attention modules — applied to music genre classification from audio representations such as mel-spectrograms, chroma features, and raw waveforms, and their extension to content-based music recommendation. The scope covers supervised genre classification on labeled datasets, attention-weight interpretability for identifying salient temporal and spectral regions, and hybrid architectures that combine attention with convolutional feature extractors. It excludes collaborative-filtering recommendation approaches that rely solely on user interaction history without audio content signals, symbolic (MIDI-based) music generation tasks, and speech or non-musical audio classification, except where used for comparative benchmarking of attention mechanisms.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of Transformer encoders applied directly to mel-spectrogram patches or pretrained audio embeddings (e.g., from wav2vec-style models) for genre and mood classification.",
        "Increasing use of hybrid CNN-attention architectures where convolutional layers extract local spectral features before self-attention models long-range temporal dependencies.",
        "Rising interest in multi-modal attention that fuses audio, lyrics, and metadata embeddings for joint genre classification and recommendation tasks.",
        "Emergence of lightweight and efficient attention variants (e.g., linear attention, local windowed attention) aimed at making Transformer-based genre classifiers feasible for real-time streaming recommendation.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of attention-based genre classifiers under label noise and inherently ambiguous or overlapping genre taxonomies common in real-world streaming catalogs.",
            "based_on_papers": ["Musical Genre Classification of Audio Signals", "Attention Is All You Need"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of attention-weight-derived similarity against learned collaborative-filtering embeddings for cold-start music recommendation.",
            "based_on_papers": ["Musical Genre Classification of Audio Signals"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarking of attention-based versus CNN/RNN genre classifiers on identical multi-genre, multi-cultural datasets with matched preprocessing and evaluation splits.",
            "based_on_papers": ["Attention Is All You Need"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Developing genre-aware attention architectures that explicitly model hierarchical or fuzzy genre taxonomies rather than assuming mutually exclusive single-label classification.",
        "Designing joint training objectives that optimize attention-based encoders simultaneously for genre classification accuracy and recommendation relevance metrics.",
        "Creating efficient, low-latency attention variants suitable for on-device or real-time streaming recommendation without sacrificing long-range dependency modeling.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "HuggingFace Datasets",
            "url": "https://huggingface.co/datasets",
            "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention-based encoder architecture that models long-range dependencies across a sequence without recurrence, applied here to sequences of spectrogram frames or audio embeddings for genre classification.",
            "verified": True,
        },
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://arxiv.org/abs/1409.1556",
            "description": "Convolutional architecture used to extract local time-frequency features from mel-spectrograms, typically combined with attention layers to capture both short-range texture and long-range structure in music audio.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Classification Accuracy",
        "Macro F1-Score",
        "Top-K Recommendation Precision@K",
        "Area Under the ROC Curve (AUC)",
    ],
    "implementation_steps": [
        "Collect and preprocess audio tracks into mel-spectrogram or chroma representations with fixed-length windows suitable for batch training.",
        "Split data into train, validation, and test sets stratified by genre label to preserve class balance across splits.",
        "Design a hybrid CNN-attention architecture where convolutional layers extract local spectral features and self-attention layers model long-range temporal dependencies across the sequence.",
        "Train the genre classifier using cross-entropy loss with label smoothing to mitigate overconfidence on ambiguous fusion-genre examples.",
        "Extract learned attention weights per track and use them to construct a content-based similarity representation for the recommendation module.",
        "Evaluate classification performance with accuracy and macro F1, and evaluate recommendation quality with top-K precision on held-out user-track interactions.",
    ],
    "unverified_suggestions": ["GTZAN Genre Collection", "Free Music Archive (FMA) dataset", "Million Song Dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates the use of attention mechanisms by grounding the objectives in the need to capture long-range temporal and spectral dependencies that CNN- and RNN-only models struggle with.",
        "The methodology specifies a concrete, reproducible pipeline from spectrogram preprocessing through hybrid CNN-attention training to recommendation evaluation.",
        "The research questions link classification and recommendation in a coherent way, treating attention weights as a shared signal rather than treating the two tasks independently.",
        "The gap analysis identifies practically relevant and underexplored issues, such as label noise from ambiguous genre taxonomies and cold-start recommendation, that extend beyond simple accuracy benchmarking.",
    ],
    "weaknesses": [
        "The proposal does not fully justify why HuggingFace Datasets is an adequate source for music audio data, since evidence sufficiency and dataset fit for genuinely large-scale, multi-cultural music catalogs remain uncertain.",
        "There is limited discussion of how genre label ambiguity and multi-label or fusion-genre tracks would be handled within a single-label classification framework.",
        "The proposal does not address the computational cost and inference latency of Transformer-based attention in enough depth given the real-time constraints of streaming recommendation systems.",
        "Baseline comparisons are limited to CNN architectures, which may understate the competitive landscape relative to modern RNN-attention hybrids or pretrained audio foundation models.",
    ],
    "missing_components": [
        "An explicit ablation plan isolating the contribution of the attention module versus the convolutional feature extractor on final classification and recommendation performance.",
        "A discussion of licensing and copyright considerations when constructing training datasets from real commercial music catalogs.",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to attention-based music genre classification and recommendation, based on the retrieved abstract." for t in titles]
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
