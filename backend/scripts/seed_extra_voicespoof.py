import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Networks for Detecting Synthetic Audio in Voice Authentication Systems"

PLANNER_JSON = {
    "research_objectives": [
        "Develop neural network countermeasures that distinguish genuine speaker audio from synthetic (text-to-speech or voice-converted) audio in voice authentication pipelines",
        "Evaluate end-to-end raw-waveform architectures against handcrafted spectral-feature classifiers for synthetic speech detection accuracy",
        "Investigate the use of self-supervised speech representations as front-end features for improving generalization to unseen synthesis and conversion methods",
        "Assess countermeasure robustness to adversarial and channel-mismatch conditions, such as codec compression and replay-through-microphone attacks",
    ],
    "research_questions": [
        "Do end-to-end raw-waveform neural networks outperform handcrafted spectral-feature classifiers for detecting synthetic audio in voice authentication?",
        "How much does incorporating self-supervised speech representations (e.g., wav2vec 2.0) improve generalization to synthesis methods not seen during training?",
        "How robust are synthetic-audio countermeasures to codec compression, channel mismatch, and replay attacks encountered in real voice authentication deployments?",
        "Can a single countermeasure model generalize across both text-to-speech and voice-conversion attack types, or are specialized detectors needed for each?",
    ],
    "keywords": [
        "synthetic speech detection",
        "voice spoofing countermeasure",
        "audio deepfake detection",
        "speaker verification anti-spoofing",
        "self-supervised speech representation",
    ],
    "research_scope": (
        "This research covers neural network methods for detecting synthetic or manipulated speech audio, including "
        "text-to-speech and voice-conversion attacks, in the context of voice authentication and speaker verification "
        "systems. It includes raw-waveform end-to-end models, spectral-feature classifiers, and self-supervised "
        "representation-based front ends, along with robustness to codec compression and replay attacks. It excludes "
        "speaker identification/diarization tasks unrelated to spoof detection and video-based deepfake detection."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of end-to-end raw-waveform architectures (e.g., RawNet2) that avoid handcrafted spectral feature extraction for spoof detection",
        "Increasing use of self-supervised speech representations, such as wav2vec 2.0, as front-end features to improve generalization to unseen synthesis methods",
        "Rising interest in cross-dataset and cross-attack generalization testing, reflecting the rapid evolution of text-to-speech and voice-conversion techniques",
        "Emerging focus on channel-mismatch and replay-attack robustness, motivated by real-world deployment conditions that differ from clean benchmark recordings",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of raw-waveform end-to-end models versus self-supervised-representation front ends under matched training data and attack conditions.",
            "based_on_papers": ["End-to-End Anti-Spoofing with RawNet2"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of countermeasure robustness to combined codec compression and replay-attack conditions simultaneously, as most studies test these factors independently.",
            "based_on_papers": ["ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: analysis of whether a single unified countermeasure model can match the accuracy of attack-type-specialized detectors across both text-to-speech and voice-conversion attacks.",
            "based_on_papers": ["ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop combined codec-compression-plus-replay robustness benchmarks that reflect realistic deployment channel conditions",
        "Establish standardized cross-attack-type generalization protocols comparing unified versus specialized countermeasure models",
        "Investigate continual or incremental learning approaches for countermeasures to adapt to newly emerging synthesis methods without full retraining",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "RawNet2",
            "url": "https://arxiv.org/abs/2011.01108",
            "description": "End-to-end raw-waveform neural network used to detect synthetic or converted speech without relying on handcrafted spectral features.",
            "verified": True,
        },
        {
            "name": "wav2vec 2.0",
            "url": "https://arxiv.org/abs/2006.11477",
            "description": "Self-supervised speech representation model used as a front end to improve generalization of synthetic-audio countermeasures to unseen attack types.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Equal error rate (EER)",
        "tandem detection cost function (t-DCF)",
        "Area under the ROC curve (AUC)",
        "Cross-attack-type generalization accuracy drop",
    ],
    "implementation_steps": [
        "Assemble labeled genuine and synthetic/converted speech corpora spanning multiple text-to-speech and voice-conversion attack types",
        "Implement a RawNet2-based end-to-end classifier and a wav2vec 2.0 self-supervised front end with a lightweight classification head",
        "Train both countermeasures on a shared attack-type distribution and evaluate cross-attack-type generalization on held-out synthesis methods",
        "Apply codec compression and simulated replay-through-microphone augmentation to the training and evaluation data to test channel-mismatch robustness",
        "Compare unified single-model countermeasures against attack-type-specialized models trained separately per synthesis or conversion method",
        "Conduct error analysis on false-accept cases to identify which synthesis methods most successfully evade each countermeasure architecture",
    ],
    "unverified_suggestions": ["ASVspoof 2019 dataset", "ASVspoof 2021 dataset", "In-the-Wild audio deepfake dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal makes a clear, testable comparison between end-to-end raw-waveform models and self-supervised-representation front ends under matched conditions",
        "Explicitly combining codec compression and replay-attack robustness testing reflects realistic deployment conditions rather than clean-benchmark-only evaluation",
        "The unified-versus-specialized countermeasure comparison addresses a practically important deployment question about maintaining a single production model",
        "Cross-attack-type generalization testing directly targets the rapid evolution of synthesis techniques that quickly outdate narrowly trained countermeasures",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) does not typically host speech anti-spoofing corpora, so ASVspoof and related datasets remain unverified suggestions requiring separate access agreements.",
        "The proposal does not address the risk that some synthesis methods used for evaluation may already be represented in the pretraining data of wav2vec 2.0, potentially inflating generalization estimates.",
        "There is limited discussion of computational latency requirements for real-time voice authentication deployment, which is an important practical constraint beyond raw detection accuracy.",
    ],
    "missing_components": [
        "An analysis of potential pretraining-data overlap between self-supervised front ends and evaluation attack types.",
        "A latency and computational-cost analysis for real-time deployment in voice authentication pipelines.",
    ],
    "novelty_score": 0.73,
    "quality_score": 0.79,
}

CONFIDENCE_JSON = {"internal_consistency": 0.85, "specificity": 0.83, "completeness": 0.87}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to detecting synthetic audio in voice authentication systems, based on the retrieved abstract." for t in titles]
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
