import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Self-Supervised Learning for Underwater Acoustic Signal Classification"

PLANNER_JSON = {
    "research_objectives": [
        "Develop self-supervised pretraining methods for underwater acoustic signal representations learned from large volumes of unlabeled passive hydrophone recordings",
        "Evaluate contrastive and predictive self-supervised objectives (e.g., contrastive predictive coding, wav2vec-style pretraining) for downstream vessel, marine mammal, and ambient-noise classification",
        "Investigate the label efficiency of self-supervised pretrained encoders when fine-tuned on small labeled underwater acoustic datasets compared to training classifiers from scratch",
        "Assess robustness of learned representations to variable ocean ambient noise conditions, sensor placement, and recording equipment differences across datasets",
    ],
    "research_questions": [
        "Can self-supervised pretraining on unlabeled hydrophone recordings learn representations that transfer effectively to vessel and marine mammal sound classification?",
        "How do contrastive objectives (e.g., contrastive predictive coding) compare to predictive/masked-prediction objectives (e.g., wav2vec 2.0-style pretraining) when adapted to underwater acoustic signals?",
        "What is the minimum amount of labeled data needed to reach competitive classification accuracy when starting from a self-supervised pretrained encoder versus training from scratch?",
        "How robust are self-supervised underwater acoustic representations to distribution shift across different recording locations, hydrophone hardware, and ambient noise profiles?",
    ],
    "keywords": [
        "self-supervised learning underwater acoustics",
        "passive sonar signal classification",
        "contrastive predictive coding audio",
        "hydrophone recording deep learning",
        "marine mammal vessel sound classification",
    ],
    "research_scope": (
        "This research focuses on self-supervised representation learning for classifying underwater acoustic "
        "signals captured by passive hydrophone sensors, including vessel noise, marine mammal vocalizations, and "
        "ambient ocean noise. The scope covers contrastive and predictive self-supervised pretraining objectives "
        "adapted from the speech and audio domain, label-efficient fine-tuning evaluation, and robustness to "
        "distribution shift across recording locations and hardware. It excludes active sonar and echo-ranging "
        "systems, underwater acoustic communication and modem signal processing, and seismic exploration "
        "applications, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adaptation of self-supervised audio pretraining objectives originally developed for speech (contrastive predictive coding, wav2vec 2.0) to underwater passive acoustic monitoring",
        "Increasing availability of open passive acoustic monitoring datasets such as DeepShip and ShipsEar, enabling larger unlabeled pretraining corpora",
        "Continued reliance on spectrogram-based CNN classifiers trained from scratch as the dominant supervised baseline in much of the existing underwater acoustics literature",
        "Rising interest in label-efficient fine-tuning to address the high cost and expertise required for expert-annotated marine mammal and vessel sound labels",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of how self-supervised pretraining objectives designed for speech transfer to the substantially different spectral and temporal characteristics of underwater vessel and marine mammal sounds.",
            "based_on_papers": ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-dataset robustness evaluation of self-supervised underwater acoustic encoders when the unlabeled pretraining corpus and the labeled fine-tuning/evaluation dataset come from different recording locations or hydrophone hardware.",
            "based_on_papers": ["Representation Learning with Contrastive Predictive Coding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: label-efficiency curves quantifying how much labeled data is saved by self-supervised pretraining specifically for underwater acoustic classification tasks, as opposed to general audio or speech benchmarks.",
            "based_on_papers": ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design underwater-acoustic-specific augmentation and pretext tasks (e.g., accounting for multipath propagation and Doppler effects) rather than directly reusing speech-domain augmentations",
        "Build cross-location, cross-hardware benchmark suites to systematically evaluate the transferability of self-supervised underwater acoustic encoders",
        "Publish standardized label-efficiency evaluation protocols for vessel and marine mammal classification to enable fair comparison across future self-supervised approaches",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Contrastive Predictive Coding (CPC)",
            "url": "https://arxiv.org/abs/1807.03748",
            "description": "Self-supervised objective that learns representations by predicting future latent codes from past context using a contrastive loss, adapted here to unlabeled hydrophone recordings.",
            "verified": True,
        },
        {
            "name": "wav2vec 2.0",
            "url": "https://arxiv.org/abs/2006.11477",
            "description": "Self-supervised speech representation framework combining a convolutional feature encoder with a masked-prediction transformer objective, adapted as an alternative pretraining approach for underwater acoustic signals.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Classification accuracy on downstream vessel/marine-mammal/ambient-noise labels",
        "Macro F1-score across acoustic classes",
        "Label-efficiency curves (accuracy versus fraction of labeled fine-tuning data)",
        "Cross-dataset transfer accuracy under recording-location and hardware shift",
    ],
    "implementation_steps": [
        "Curate an unlabeled pretraining corpus from open passive acoustic monitoring archives, segmenting continuous hydrophone recordings into fixed-length clips",
        "Adapt contrastive predictive coding and wav2vec 2.0-style pretraining objectives to the spectral and temporal characteristics of underwater acoustic signals",
        "Pretrain encoders on the unlabeled corpus, then attach a lightweight classification head fine-tuned on labeled vessel, marine mammal, and ambient-noise datasets",
        "Evaluate downstream classification accuracy under multiple label budgets (e.g., 1%, 10%, 100% of labeled data) to quantify label-efficiency gains from pretraining",
        "Test cross-dataset generalization by pretraining on one recording location/hardware setup and fine-tuning/evaluating on a distinct dataset such as DeepShip or ShipsEar",
        "Compare against spectrogram-based CNN classifiers trained from scratch as the supervised baseline",
    ],
    "unverified_suggestions": [
        "DeepShip underwater vessel acoustic dataset",
        "ShipsEar underwater vessel noise dataset",
        "Watkins Marine Mammal Sound Database",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal correctly identifies that speech-domain self-supervised objectives cannot be assumed to transfer directly to underwater acoustics, and builds this question directly into the research plan rather than assuming transferability",
        "Including label-efficiency curves as a primary evaluation output directly tests the core motivation for self-supervised pretraining rather than reporting only a single fine-tuned accuracy figure",
        "The cross-dataset generalization test (pretraining on one recording setup, evaluating on another) is a realistic and informative robustness check given known variability in ocean ambient noise and hydrophone hardware",
        "Comparing two distinct self-supervised paradigms (contrastive predictive coding and masked-prediction wav2vec 2.0) gives a more informative picture than committing to a single pretraining objective",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how underwater-acoustic-specific augmentations (e.g., simulating multipath propagation or Doppler shift) will be designed, despite noting that speech-domain augmentations may not transfer directly",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host passive acoustic monitoring recordings, so the project depends on the unverified DeepShip, ShipsEar, and Watkins dataset suggestions, which differ in recording conditions and label taxonomies that may complicate cross-dataset comparison",
        "The plan does not address a genuinely important methodological consideration for this domain: ambient ocean noise level varies significantly with weather, shipping traffic, and biological activity, and this potential confound is not discussed as a control variable in the classification evaluation",
    ],
    "missing_components": [
        "A plan for controlling or stratifying evaluation by ambient noise level to avoid confounding classification accuracy with easy versus hard acoustic conditions",
        "A discussion of computational and data-scale requirements needed for self-supervised pretraining to reach representation quality competitive with supervised baselines",
    ],
    "novelty_score": 0.75,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.80, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to self-supervised learning for underwater acoustic signal classification, based on the retrieved abstract." for t in titles]
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
