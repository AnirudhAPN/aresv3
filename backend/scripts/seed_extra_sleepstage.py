import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Sleep Stage Classification from Wearable EEG Signals"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning model that classifies sleep stages (wake, N1, N2, N3, REM) from reduced-channel EEG signals captured by consumer wearable devices",
        "Evaluate CNN and recurrent/transformer temporal architectures for their ability to model sleep-stage transitions across full-night recordings",
        "Investigate the accuracy tradeoff between clinical-grade multi-channel polysomnography (PSG) EEG and single/dual-channel wearable EEG for automated staging",
        "Assess model robustness to inter-subject variability and to signal artifacts such as motion and electrode displacement typical of at-home wearable recordings",
    ],
    "research_questions": [
        "How much does classification accuracy degrade when moving from multi-channel clinical PSG EEG to single or dual-channel wearable EEG signals?",
        "Which temporal modeling architectures, including CNN, bidirectional LSTM, and transformer, best capture the sequential structure of sleep-stage transitions across a full night?",
        "How robust are wearable EEG sleep-stage classifiers to inter-subject variability without subject-specific calibration?",
        "What is the impact of motion and electrode-displacement artifacts on staging accuracy, and which preprocessing or artifact-rejection strategies mitigate it most effectively?",
    ],
    "keywords": [
        "sleep stage classification deep learning",
        "wearable EEG sleep staging",
        "polysomnography automated scoring",
        "single-channel EEG sleep",
        "REM detection neural network",
    ],
    "research_scope": (
        "This research covers deep learning models for automated sleep-stage classification from EEG signals, "
        "comparing clinical multi-channel polysomnography against reduced-channel consumer wearable EEG, "
        "including temporal sequence modeling and artifact robustness. It treats non-EEG wearable modalities "
        "such as actigraphy or photoplethysmography only as auxiliary signals, not the primary focus, and "
        "excludes clinical diagnosis of specific sleep disorders beyond stage classification itself."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Adoption of CNN-based feature extractors combined with bidirectional RNN or transformer sequence models to capture whole-night sleep-stage transition structure, as in DeepSleepNet and SeqSleepNet",
        "Growing focus on single- or dual-channel EEG staging to match the sensor constraints of consumer-grade wearable devices",
        "Use of self-supervised pretraining on large unlabeled PSG corpora to improve staging accuracy on smaller labeled wearable datasets",
        "Increasing attention to domain adaptation techniques addressing the distribution shift between clinical PSG electrode placement and wearable electrode positions",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled, matched evaluation protocol quantifying the accuracy gap between multi-channel clinical PSG and single/dual-channel wearable EEG.",
            "based_on_papers": ["DeepSleepNet: A Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of inter-subject generalization without per-subject calibration, since many reported results rely on subject-dependent train/test splits.",
            "based_on_papers": ["SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: explicit robustness evaluation to motion and electrode-displacement artifacts typical of at-home wearable use, since most benchmarks rely on clean clinical-lab recordings.",
            "based_on_papers": ["DeepSleepNet: A Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Construct paired PSG/wearable-EEG benchmarks enabling direct, matched accuracy-gap quantification",
        "Develop and standardize subject-independent training protocols with leave-one-subject-out validation",
        "Design artifact-robust preprocessing and augmentation pipelines that simulate at-home motion and electrode-displacement conditions during training",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "DeepSleepNet",
            "url": "https://arxiv.org/abs/1703.04046",
            "description": "CNN feature extractor combined with bidirectional LSTM sequence modeling for automatic single-channel EEG sleep staging, used as the primary architecture under evaluation.",
            "verified": True,
        },
        {
            "name": "Transformer encoder for sequence classification",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention-based sequence architecture adapted to model transitions across sequential 30-second EEG epochs as an alternative to recurrent temporal modeling.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Overall accuracy",
        "Macro-averaged F1-score",
        "Cohen's kappa agreement with expert scorers",
        "Per-stage sensitivity and specificity, especially for N1 and REM",
    ],
    "implementation_steps": [
        "Assemble paired clinical PSG and wearable-compatible reduced-channel EEG recordings with expert-scored sleep-stage labels",
        "Preprocess EEG signals into standardized 30-second epochs with consistent filtering and normalization across channel configurations",
        "Implement and train CNN+bidirectional-LSTM (DeepSleepNet-style) and transformer-based sequence models on both full-channel and reduced-channel inputs",
        "Evaluate accuracy degradation from multi-channel to single/dual-channel input under matched subject splits",
        "Perform leave-one-subject-out cross-validation to assess inter-subject generalization without per-subject calibration",
        "Inject simulated motion and electrode-displacement artifacts into wearable-channel recordings and evaluate model robustness with and without artifact-rejection preprocessing",
    ],
    "unverified_suggestions": ["Sleep-EDF Database", "MASS (Montreal Archive of Sleep Studies)", "SHHS (Sleep Heart Health Study)"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly quantifies a practically important tradeoff, clinical PSG accuracy versus wearable EEG accuracy, rather than assuming wearable performance parity",
        "Leave-one-subject-out validation is explicitly built into the methodology, addressing a common and consequential evaluation weakness in the sleep-staging literature",
        "The artifact-robustness evaluation targets a realistic deployment condition (at-home motion and electrode displacement) rather than only clean lab data",
        "Evaluation metrics appropriately include per-stage sensitivity for N1 and REM, the stages most prone to classification error and clinically significant",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how paired PSG/wearable recordings will be obtained, since simultaneous dual-modality recordings are relatively rare and may require a dedicated data collection effort beyond existing public datasets",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host raw polysomnography EEG datasets such as Sleep-EDF or SHHS, so those remain unverified suggestions requiring separate access via PhysioNet or NSRR",
        "The plan does not address how inter-scorer disagreement among human experts, a known source of label noise in sleep staging, will be handled when computing ground-truth agreement metrics",
    ],
    "missing_components": [
        "A plan for sourcing or constructing paired PSG/wearable-EEG recordings, given the rarity of existing simultaneous dual-modality datasets",
        "A discussion of inter-scorer label noise and how it will be factored into agreement metric interpretation",
    ],
    "novelty_score": 0.62,
    "quality_score": 0.71,
}

CONFIDENCE_JSON = {"internal_consistency": 0.77, "specificity": 0.75, "completeness": 0.80}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning sleep stage classification from wearable EEG signals, based on the retrieved abstract." for t in titles]
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
