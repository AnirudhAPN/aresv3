import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Lip Reading from Silent Video"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning model for word- and sentence-level lip reading from silent video, without relying on any audio input",
        "Evaluate spatiotemporal architectures combining 3D convolutional visual front-ends with transformer/Conformer temporal encoders for visual speech recognition accuracy",
        "Investigate cross-speaker generalization of lip reading models trained on datasets with limited speaker diversity",
        "Assess robustness of lip reading models to variation in lighting conditions, head pose, and video resolution typical of in-the-wild footage",
    ],
    "research_questions": [
        "Can transformer-based visual speech recognition models achieve sentence-level lip reading accuracy competitive with audio-based ASR under ideal recording conditions?",
        "How does cross-speaker generalization degrade when training on datasets with limited speaker diversity, such as LRS2/LRS3?",
        "What is the impact of head pose and lighting variation on lip reading accuracy, and can targeted data augmentation mitigate this degradation?",
        "How do self-supervised audio-visual pretraining approaches (e.g., AV-HuBERT) compare to purely visual pipelines when only silent video is available at inference time?",
    ],
    "keywords": [
        "lip reading deep learning",
        "visual speech recognition",
        "silent speech video transformer",
        "lipreading LRS3 dataset",
        "audio-visual self-supervised speech recognition",
    ],
    "research_scope": (
        "This research focuses on visual-only speech recognition (lip reading) from silent talking-face video, "
        "spanning word-level and sentence-level recognition tasks. It covers spatiotemporal deep learning "
        "architectures (3D-CNN visual front-ends, Conformer/transformer temporal encoders), self-supervised "
        "audio-visual pretraining used strictly as a representation-learning tool prior to audio-absent inference, "
        "and robustness evaluation under lighting, pose, and resolution variation. The scope excludes audio-visual "
        "speech recognition systems that require audio at inference time, speaker identification/verification from "
        "lip movement, and multi-speaker overlapping-speech separation, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Shift from CNN+LSTM lip reading pipelines toward transformer and Conformer-based visual speech recognition architectures",
        "Growing use of self-supervised audio-visual pretraining (e.g., AV-HuBERT) to learn visual speech representations without requiring large amounts of labeled transcripts",
        "Adoption of large-scale in-the-wild datasets (LRS2, LRS3, LRW) for training, despite these datasets having limited speaker diversity relative to real-world deployment settings",
        "Increasing interest in lip reading as assistive technology for hearing-impaired users and as a modality for silent-speech human-computer interaction",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: rigorous cross-speaker generalization benchmarking that isolates speaker-identity effects from vocabulary/content effects on lip reading accuracy.",
            "based_on_papers": ["LRS3-TED: a large-scale dataset for visual speech recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of lighting and head-pose robustness under controlled, parametric perturbation, as opposed to relying only on the natural diversity present in in-the-wild datasets.",
            "based_on_papers": ["Deep Audio-Visual Speech Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of purely visual lip reading pipelines against self-supervised audio-visual pretrained models specifically under audio-absent (silent) inference conditions.",
            "based_on_papers": ["Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop benchmark protocols with held-out speaker splits explicitly designed to disentangle speaker identity from lexical content generalization",
        "Design controlled perturbation test suites (synthetic lighting/pose augmentation) for systematic robustness evaluation of lip reading models",
        "Investigate distillation of audio-visual pretrained representations into purely visual student models optimized for audio-absent deployment",
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
            "name": "3D Convolutional Neural Network (3D-CNN) visual front-end",
            "url": "https://arxiv.org/abs/1611.01599",
            "description": "Spatiotemporal convolutional network applied to cropped mouth-region video frames to extract low-level visual speech features prior to temporal sequence modeling.",
            "verified": True,
        },
        {
            "name": "Conformer",
            "url": "https://arxiv.org/abs/2005.08100",
            "description": "Convolution-augmented transformer architecture used as the temporal encoder over 3D-CNN visual features to model both local and long-range dependencies in the lip movement sequence.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Word Error Rate (WER)",
        "Character Error Rate (CER)",
        "Sentence-level accuracy",
        "BLEU score",
    ],
    "implementation_steps": [
        "Collect and preprocess LRS2/LRS3 video corpora, including face detection and lip region-of-interest cropping",
        "Extract per-frame lip region sequences and normalize for scale, rotation, and illumination variation",
        "Train a 3D-CNN visual front-end to extract spatiotemporal features from lip region sequences",
        "Stack a Conformer/transformer temporal encoder with a CTC or sequence-to-sequence decoder on top of the visual front-end",
        "Apply data augmentation (synthetic lighting changes, pose jitter, resolution downsampling) to improve robustness",
        "Evaluate cross-speaker generalization using held-out speaker splits and compute WER, CER, and sentence-level accuracy",
    ],
    "unverified_suggestions": ["LRS3-TED", "LRW (Lip Reading in the Wild)", "GRID corpus"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal isolates audio-absent inference as an explicit design constraint, which is often glossed over in audio-visual speech recognition literature",
        "Combining a 3D-CNN visual front-end with a Conformer temporal encoder reflects current state-of-the-art architecture choices for the task",
        "The cross-speaker generalization objective directly targets a well-known and practically important limitation of existing lip reading benchmarks",
        "The robustness evaluation plan explicitly separates lighting and pose perturbation testing from ordinary in-the-wild dataset diversity, enabling more controlled analysis",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how held-out speaker splits will control for vocabulary overlap, risking conflation of speaker generalization with simple content memorization effects",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host video-based lip reading corpora, so feasibility depends entirely on the unverified, domain-specific datasets listed",
        "The plan does not address how homophenes (visually indistinguishable phonemes, such as /p/, /b/, /m/) will be handled, which is a well-known fundamental limitation of lip reading that architecture changes alone cannot fully resolve",
    ],
    "missing_components": [
        "An explicit discussion of homophene ambiguity and how language-model-based decoding will be used to disambiguate visually indistinguishable phoneme sequences",
        "A plan for obtaining informed consent and addressing privacy considerations given the sensitive nature of silent lip reading technology",
    ],
    "novelty_score": 0.65,
    "quality_score": 0.78,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning for automated lip reading from silent video, based on the retrieved abstract." for t in titles]
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
