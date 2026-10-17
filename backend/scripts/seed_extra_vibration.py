import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Self-Supervised Learning for Anomaly Detection in Industrial Vibration Sensor Data"

PLANNER_JSON = {
    "research_objectives": [
        "Develop self-supervised representation learning methods that learn normal-operation patterns from unlabeled industrial vibration sensor data for downstream anomaly detection",
        "Evaluate contrastive and time-series-specific self-supervised pretraining objectives for their ability to produce representations that separate healthy from faulty machine states",
        "Investigate how much labeled fault data is needed to fine-tune a self-supervised pretrained model to competitive fault-detection accuracy, compared to fully supervised baselines",
        "Assess the transferability of self-supervised vibration representations across different machine types and sensor placements without full retraining",
    ],
    "research_questions": [
        "Do self-supervised pretrained representations of vibration signals improve downstream anomaly detection accuracy compared to training a supervised model from scratch on limited labeled fault data?",
        "How do contrastive learning objectives compare to time-series-specific self-supervised objectives (e.g., temporal contrastive coding) for vibration anomaly detection?",
        "What is the minimum amount of labeled fault data required for a self-supervised pretrained model to match fully supervised fault-detection accuracy?",
        "Can self-supervised vibration representations learned on one machine type transfer effectively to a different machine type or sensor placement?",
    ],
    "keywords": [
        "self-supervised time series representation",
        "vibration anomaly detection",
        "industrial predictive maintenance",
        "contrastive learning sensor data",
        "bearing fault detection",
    ],
    "research_scope": (
        "This research covers self-supervised representation learning methods applied to industrial vibration sensor "
        "data for anomaly and fault detection, including contrastive and time-series-specific pretraining objectives, "
        "label-efficiency evaluation, and cross-machine transferability. It excludes acoustic emission or thermal "
        "sensing modalities, purely supervised fault classification without a self-supervised pretraining component, "
        "and non-industrial (e.g., structural health monitoring of buildings) vibration analysis applications."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of contrastive self-supervised objectives adapted from computer vision to time-series vibration signals for label-efficient anomaly detection",
        "Increasing interest in time-series-specific self-supervised methods that exploit temporal structure rather than generic augmentation-based contrastive learning",
        "Rising adoption of pretrain-then-fine-tune pipelines that use large amounts of unlabeled normal-operation data followed by fine-tuning on a small labeled fault set",
        "Emerging focus on cross-machine and cross-domain transfer of learned vibration representations to reduce per-machine data collection costs",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of generic contrastive learning (e.g., SimCLR-style) versus time-series-specific self-supervised objectives on the same vibration fault-detection benchmark.",
            "based_on_papers": ["A Simple Framework for Contrastive Learning of Visual Representations"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic label-efficiency curves quantifying how much labeled fault data is needed for self-supervised pretrained models to match fully supervised baselines.",
            "based_on_papers": ["TS2Vec: Towards Universal Representation of Time Series"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of cross-machine-type transfer of self-supervised vibration representations under differing sensor placements and sampling rates.",
            "based_on_papers": ["TS2Vec: Towards Universal Representation of Time Series"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish standardized label-efficiency benchmarks comparing self-supervised pretrained and fully supervised vibration fault detectors",
        "Develop cross-machine transfer benchmarks spanning multiple bearing/gearbox types and sensor configurations",
        "Study combinations of generic contrastive and time-series-specific self-supervised objectives to capture complementary signal structure",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "TS2Vec",
            "url": "https://arxiv.org/abs/2106.10466",
            "description": "Time-series-specific self-supervised representation learning method used to pretrain vibration signal encoders on unlabeled normal-operation data.",
            "verified": True,
        },
        {
            "name": "SimCLR",
            "url": "https://arxiv.org/abs/2002.05709",
            "description": "Generic contrastive self-supervised learning framework adapted to vibration signal windows, used as a comparison pretraining objective against time-series-specific methods.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Anomaly detection AUC-ROC on held-out fault data",
        "F1-score for fault classification after fine-tuning",
        "Label-efficiency curve (accuracy versus fraction of labeled fault data used)",
        "Cross-machine transfer accuracy drop",
    ],
    "implementation_steps": [
        "Collect unlabeled normal-operation vibration signals and a smaller labeled set of known fault conditions from industrial machinery (e.g., bearings, gearboxes)",
        "Pretrain encoders using TS2Vec and SimCLR-style contrastive objectives on the unlabeled normal-operation data",
        "Fine-tune each pretrained encoder on varying fractions of the labeled fault data to construct label-efficiency curves",
        "Implement a fully supervised baseline trained from scratch on the same labeled fault data for direct comparison",
        "Evaluate cross-machine transfer by pretraining on one machine type and fine-tuning/evaluating on a different machine type or sensor placement",
        "Conduct ablation studies isolating the effect of pretraining objective choice and pretraining data volume on downstream fault-detection accuracy",
    ],
    "unverified_suggestions": ["NASA IMS bearing dataset", "CWRU (Case Western Reserve University) bearing fault dataset", "MIMII industrial machine sound dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets label efficiency, a genuinely important practical constraint in industrial settings where fault labels are scarce and expensive to obtain",
        "Comparing generic contrastive and time-series-specific self-supervised objectives on a shared benchmark provides a clear, reproducible experimental contribution",
        "The cross-machine transfer evaluation addresses a realistic deployment scenario where collecting fresh labeled data for every new machine is impractical",
        "Constructing explicit label-efficiency curves, rather than a single fixed-label-fraction result, gives a more complete picture of the self-supervised pretraining benefit",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) does not typically host raw industrial vibration time-series benchmarks, so NASA IMS, CWRU, and MIMII remain unverified suggestions requiring separate download and licensing checks.",
        "The proposal does not address how differing fault severities and progressive degradation (rather than binary healthy/faulty labels) would be incorporated into the anomaly detection framing.",
        "There is limited discussion of sampling-rate and sensor-type mismatches across the different bearing/gearbox datasets, which could confound the cross-machine transfer results if not carefully controlled.",
    ],
    "missing_components": [
        "A framing of fault severity or progressive degradation levels, rather than a purely binary healthy/faulty label scheme.",
        "A control for sampling-rate and sensor-type differences when evaluating cross-machine transfer.",
    ],
    "novelty_score": 0.66,
    "quality_score": 0.73,
}

CONFIDENCE_JSON = {"internal_consistency": 0.77, "specificity": 0.78, "completeness": 0.81}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to self-supervised anomaly detection in industrial vibration sensor data, based on the retrieved abstract." for t in titles]
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
