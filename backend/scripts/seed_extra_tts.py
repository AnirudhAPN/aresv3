import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Text-to-Speech Synthesis for Low-Resource Language Accessibility"

PLANNER_JSON = {
    "research_objectives": [
        "Develop neural text-to-speech (TTS) architectures that produce intelligible, natural-sounding speech for languages with limited transcribed audio data",
        "Investigate transfer learning and cross-lingual adaptation techniques that let high-resource TTS models bootstrap low-resource voices",
        "Evaluate data-efficient training strategies, including self-supervised pretraining and voice cloning, for reducing the amount of paired text-audio data required",
        "Assess the real-world accessibility impact of low-resource TTS systems for speakers of underserved languages",
    ],
    "research_questions": [
        "How much paired text-audio data is required for a neural TTS model to reach acceptable naturalness and intelligibility in a low-resource language?",
        "Which cross-lingual transfer learning strategies most effectively reuse phonetic and prosodic knowledge from high-resource source languages?",
        "Can self-supervised speech representations reduce the reliance on large labeled datasets for low-resource TTS training?",
        "What evaluation protocols best capture the accessibility benefits of low-resource TTS for real users, beyond standard MOS naturalness scores?",
    ],
    "keywords": [
        "low-resource text-to-speech",
        "cross-lingual voice transfer",
        "neural speech synthesis",
        "multilingual TTS",
        "speech accessibility",
    ],
    "research_scope": (
        "This research focuses on neural text-to-speech synthesis methods tailored to languages that lack the "
        "large, high-quality paired text-audio corpora typically used to train state-of-the-art systems such as "
        "Tacotron 2 or FastSpeech 2. The scope spans data-efficient acoustic modeling, cross-lingual and "
        "cross-speaker transfer learning, self-supervised pretraining on unlabeled speech, and vocoder choices "
        "suited to constrained compute environments. It also considers the downstream accessibility impact of "
        "these systems, including screen-reader and assistive-technology use cases for linguistic communities "
        "that are otherwise excluded from modern voice interfaces."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread use of transformer-based acoustic models (e.g., FastSpeech 2, Tacotron 2) paired with neural vocoders such as WaveGlow or HiFi-GAN for high-resource languages",
        "Growing adoption of self-supervised speech representations (e.g., wav2vec 2.0-style pretraining) to reduce labeled-data requirements",
        "Increasing interest in cross-lingual and multilingual TTS models that share acoustic decoders across many languages to enable zero-shot or few-shot voice synthesis",
        "Use of voice cloning and speaker-adaptation techniques to synthesize new voices from only a few minutes of target-language audio",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized, language-agnostic benchmarks for measuring intelligibility and naturalness specifically in tonal and low-resource languages, as opposed to English-centric MOS evaluation.",
            "based_on_papers": ["FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of how phoneme inventory mismatch between source and target languages affects the quality of cross-lingual transfer in TTS acoustic models.",
            "based_on_papers": ["Transfer Learning from Speech Synthesis to Voice Conversion with Non-Parallel Training Data"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: deployment-oriented evaluation of low-resource TTS systems on low-power or offline devices used in the accessibility contexts where these languages are most spoken.",
            "based_on_papers": ["HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build open, community-sourced low-resource speech corpora paired with standardized transcription protocols to enable reproducible benchmarking",
        "Develop unified multilingual acoustic models that share phonetic embeddings across typologically diverse languages to improve zero-shot transfer",
        "Design lightweight vocoders and acoustic models optimized for on-device inference in low-connectivity accessibility settings",
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
            "description": "Self-attention-based sequence model that underlies most modern TTS acoustic models, providing the encoder-decoder backbone used to map phoneme sequences to mel-spectrogram frames.",
            "verified": True,
        },
        {
            "name": "FastSpeech 2",
            "url": "https://arxiv.org/abs/2006.04558",
            "description": "Non-autoregressive TTS acoustic model that directly predicts duration, pitch, and energy, enabling faster and more stable training than autoregressive models like Tacotron 2, which is useful when only limited target-language data is available for fine-tuning.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Mean Opinion Score (MOS) for perceived naturalness",
        "Word Error Rate (WER) from an ASR system transcribing synthesized speech, as a proxy for intelligibility",
        "Mel-Cepstral Distortion (MCD) between synthesized and reference audio",
        "Speaker similarity score for voice-cloned or adapted low-resource voices",
    ],
    "implementation_steps": [
        "Assemble and clean a paired text-audio corpus for the target low-resource language, including phoneme-level transcription where available",
        "Pretrain or select a self-supervised speech representation model on available unlabeled audio for the target language or related languages",
        "Fine-tune a multilingual acoustic model (e.g., a FastSpeech 2 variant) using transfer learning from a high-resource source language",
        "Train or adapt a neural vocoder to convert predicted mel-spectrograms into waveform audio",
        "Run objective evaluation (WER, MCD, speaker similarity) and human MOS studies with native speakers of the target language",
        "Package the resulting model for lightweight, offline-capable deployment in accessibility tools such as screen readers",
    ],
    "unverified_suggestions": [
        "Explore diffusion-based vocoders as a higher-fidelity alternative to GAN-based vocoders for low-resource voices",
        "Investigate crowdsourced data collection via mobile apps to rapidly grow low-resource speech corpora",
        "Consider few-shot meta-learning approaches for rapid adaptation to new low-resource languages with minimal data",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research objectives that connect technical TTS methods to a concrete accessibility outcome",
        "Methodology combines established acoustic modeling (FastSpeech 2) with practical transfer-learning strategies suited to low-resource settings",
        "Evaluation plan pairs objective metrics (WER, MCD) with human MOS studies, giving both scalable and perceptually grounded assessment",
        "Future directions address real deployment constraints, such as on-device inference, rather than staying purely academic",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is fairly small for a topic that spans acoustic modeling, transfer learning, and accessibility evaluation, so some conclusions may be under-supported",
        "Dataset fit is a concern: the listed HuggingFace Datasets resource is generic and the proposal does not verify that specific low-resource-language speech corpora actually exist there",
        "The gap analysis does not deeply engage with linguistic typology (e.g., tonal languages, agglutinative morphology) that could affect phoneme modeling choices",
        "No explicit plan for handling code-switching or dialectal variation, which is common in many low-resource language communities",
    ],
    "missing_components": [
        "A concrete data collection or partnership plan for obtaining target-language audio when public corpora are insufficient",
        "A discussion of ethical and community-consent considerations when recording and distributing speech data for underserved language groups",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to low-resource neural text-to-speech synthesis, based on the retrieved abstract." for t in titles]
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
