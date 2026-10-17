import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer Architectures for Automatic Speech Recognition in Noisy Environments"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate how well Transformer-based and Conformer-based acoustic models maintain recognition accuracy under realistic additive and reverberant noise conditions compared to clean-speech baselines",
        "Design and benchmark noise-robustness strategies (multi-condition training, front-end denoising, self-supervised pretraining) applied to Transformer ASR architectures",
        "Quantify the contribution of convolutional sub-sampling and relative positional encoding schemes to robustness against non-stationary background noise",
        "Identify architectural and training modifications that reduce the word error rate (WER) degradation gap between clean and noisy test conditions for end-to-end Transformer ASR systems",
    ],
    "research_questions": [
        "How does the recognition accuracy of Transformer-based end-to-end ASR models degrade as a function of signal-to-noise ratio (SNR) compared to hybrid DNN-HMM and RNN-based systems?",
        "Which noise-robustness techniques (multi-condition data augmentation, SpecAugment, self-supervised pretraining such as wav2vec 2.0) yield the largest WER reduction under low-SNR and reverberant conditions?",
        "Does combining convolutional feature extraction with self-attention (as in the Conformer architecture) provide measurable robustness gains over pure self-attention Transformers in noisy environments?",
        "What is the computational and latency tradeoff of noise-robust Transformer ASR architectures relative to their accuracy gains, particularly for streaming or on-device deployment?",
    ],
    "keywords": [
        "transformer speech recognition",
        "noise-robust ASR",
        "conformer architecture",
        "self-supervised speech pretraining",
        "multi-condition training",
    ],
    "research_scope": (
        "This research focuses on end-to-end Transformer and Conformer architectures for automatic speech "
        "recognition under additive noise, reverberation, and channel distortion conditions typical of "
        "real-world deployment (e.g., in-car, crowded public spaces, far-field microphones). It covers "
        "encoder-decoder and CTC/RNN-T hybrid Transformer variants, data augmentation and multi-condition "
        "training strategies, and self-supervised pretraining approaches (e.g., wav2vec 2.0, HuBERT) used to "
        "improve robustness. The scope excludes speaker diarization, multi-speaker source separation as a "
        "standalone task, and non-neural signal processing front ends studied in isolation from the "
        "recognition model, which are treated as out of scope for this analysis."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of Conformer architectures that combine convolutional local feature extraction with self-attention to capture both local and global acoustic context",
        "Widespread use of self-supervised pretraining (wav2vec 2.0, HuBERT) on large unlabeled speech corpora followed by fine-tuning on smaller labeled, noise-augmented datasets",
        "Increasing reliance on multi-condition training and SpecAugment-style spectral masking to simulate noisy and reverberant conditions during training rather than relying solely on explicit denoising front ends",
        "Interest in streaming-compatible Transformer variants (chunk-based or Emformer-style attention) that preserve noise robustness while meeting low-latency requirements for on-device ASR",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of Transformer versus Conformer encoders under a unified multi-condition training protocol across a shared set of noise types and SNR levels.",
            "based_on_papers": ["Attention Is All You Need"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of how self-supervised pretraining corpus composition (clean vs. noisy speech) affects downstream ASR robustness after fine-tuning on limited labeled noisy data.",
            "based_on_papers": ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", "Conformer: Convolution-augmented Transformer for Speech Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: joint evaluation of streaming latency constraints and noise-robustness accuracy tradeoffs for chunk-based attention mechanisms in real-time ASR deployment.",
            "based_on_papers": ["Conformer: Convolution-augmented Transformer for Speech Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop unified noisy-speech benchmarks that evaluate Transformer and Conformer ASR architectures consistently across matched noise types, SNR ranges, and reverberation profiles",
        "Investigate noise-aware self-supervised pretraining objectives that explicitly incorporate diverse acoustic conditions rather than primarily clean or lightly augmented speech",
        "Explore adaptive streaming attention mechanisms that dynamically trade off latency and robustness based on estimated ambient noise level",
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
            "name": "Transformer (Attention Is All You Need)",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention-based encoder-decoder architecture that replaces recurrence with multi-head attention, serving as the foundational sequence model underlying end-to-end Transformer ASR systems.",
            "verified": True,
        },
        {
            "name": "Conformer",
            "url": "https://arxiv.org/abs/2005.08100",
            "description": "Convolution-augmented Transformer architecture that interleaves convolutional modules with self-attention layers to jointly model local and global acoustic dependencies, used as the primary noise-robust encoder under evaluation.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Word error rate (WER) across varying SNR levels",
        "Character error rate (CER)",
        "Real-time factor (RTF) for streaming latency",
        "Relative WER degradation from clean to noisy test conditions",
    ],
    "implementation_steps": [
        "Curate clean and noisy speech corpora, synthesizing multi-condition training data via additive noise mixing and simulated room impulse response convolution across a range of target SNRs",
        "Extract log-mel filterbank features and apply SpecAugment-style time and frequency masking during training to improve robustness to spectral distortion",
        "Pretrain or fine-tune self-supervised speech representations (e.g., wav2vec 2.0 style) on unlabeled multi-condition audio before attaching a CTC/RNN-T decoding head",
        "Train and compare Transformer and Conformer encoder variants under identical multi-condition training and decoding configurations",
        "Evaluate each architecture on held-out clean and noisy test sets, computing WER, CER, and real-time factor across SNR buckets and noise types",
        "Perform ablations isolating the effect of convolutional sub-sampling, positional encoding scheme, and pretraining corpus composition on noise robustness",
    ],
    "unverified_suggestions": ["LibriSpeech", "CHiME-4", "VoiceBank-DEMAND"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly articulates a focused research question comparing Transformer and Conformer noise robustness under controlled multi-condition training",
        "The methodology specifies a concrete, reproducible experimental pipeline spanning data augmentation, self-supervised pretraining, and architecture comparison",
        "Evaluation metrics are appropriate for noisy ASR, combining accuracy measures (WER, CER) with a practical deployment metric (real-time factor)",
        "The identified gaps are specific and build logically on limitations of the cited Transformer and Conformer speech recognition literature",
    ],
    "weaknesses": [
        "The proposal does not fully specify how reverberation profiles will be sampled or validated as representative of real-world far-field conditions, leaving part of the noise simulation pipeline underspecified",
        "Evidence sufficiency is limited: only a small set of retrieved papers directly benchmark Transformer versus Conformer robustness under matched noisy conditions, so conclusions should be treated as preliminary until validated on the proposed unified benchmark",
        "Dataset fit caveats apply since the whitelisted dataset repository (OpenML) may not directly host the specialized noisy-speech benchmarks implied by the unverified suggestions",
        "The plan does not address how streaming latency constraints interact with the accuracy gains from self-supervised pretraining, leaving a key deployment tradeoff unresolved",
    ],
    "missing_components": [
        "A clear baseline comparison protocol against non-Transformer noise-robust ASR methods such as hybrid DNN-HMM systems with explicit denoising front ends",
        "An explicit plan for statistical significance testing of WER differences across noise types and SNR levels",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to noise-robust Transformer speech recognition architectures, based on the retrieved abstract." for t in titles]
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
