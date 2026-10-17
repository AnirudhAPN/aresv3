import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer-Based Anomaly Detection in Time-Series Sensor Data for Predictive Aviation Maintenance"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based anomaly detection models for multivariate time-series sensor data from aircraft engines and onboard systems",
        "Evaluate self-attention architectures against LSTM-autoencoder and threshold-based baselines for early detection of degradation patterns preceding maintenance events",
        "Investigate unsupervised and semi-supervised anomaly detection approaches given the rarity of labeled failure events in aviation sensor data",
        "Assess attention-based interpretability methods for explaining flagged anomalies to support maintenance engineer decision-making and regulatory trust",
    ],
    "research_questions": [
        "Can transformer-based anomaly detectors identify early degradation signatures in multivariate sensor data earlier than threshold-based or LSTM-autoencoder baselines?",
        "How do unsupervised transformer anomaly detection methods perform given the extreme label scarcity of actual in-service failure events?",
        "Which attention-based interpretability techniques best explain which sensor channels and timesteps drove an anomaly flag, in terms usable by maintenance engineers?",
        "How well do models trained on one aircraft or engine fleet generalize to a different fleet with different sensor configurations and operating conditions?",
    ],
    "keywords": [
        "transformer anomaly detection time series",
        "predictive maintenance aviation sensor data",
        "multivariate time series anomaly detection",
        "aircraft engine degradation prediction",
        "self-attention time series forecasting",
    ],
    "research_scope": (
        "This research focuses on transformer-based anomaly detection and degradation forecasting applied to "
        "multivariate time-series sensor data from aircraft engines and mechanical systems, covering unsupervised "
        "and semi-supervised training regimes, comparison against LSTM-autoencoder and threshold-based baselines, "
        "and attention-based interpretability analysis. It uses simulated degradation benchmarks and, where "
        "available, real fleet telemetry patterns as reference. The scope excludes structural airframe fatigue "
        "modeling, air traffic control or flight scheduling optimization, and full maintenance workflow/logistics "
        "systems, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of transformer architectures for multivariate time-series anomaly detection, extending beyond LSTM-autoencoder baselines",
        "Use of the NASA C-MAPSS turbofan degradation simulation dataset as a standard benchmark for predictive maintenance research",
        "Increasing focus on unsupervised and self-supervised anomaly detection given the scarcity of labeled failure events in real aviation fleets",
        "Interest in attention-weight-based interpretability to support explainable predictive maintenance for regulatory and safety-critical decision contexts",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of transformer anomaly detectors on real in-service fleet sensor data rather than only simulated degradation benchmarks such as C-MAPSS, which may not capture real operational variability.",
            "based_on_papers": ["Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of cross-fleet generalization when sensor configurations or aircraft/engine types differ from the training fleet.",
            "based_on_papers": ["Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative evaluation of attention-based interpretability faithfulness, i.e., whether highlighted sensor channels and timesteps truly correspond to the actual root-cause degradation mechanism, rather than qualitative visualization alone.",
            "based_on_papers": ["Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish evaluation protocols using real, anonymized in-service fleet telemetry alongside simulated benchmarks to test generalization beyond C-MAPSS-style data",
        "Develop domain-adaptation techniques for transferring anomaly detection models across aircraft/engine types with differing sensor configurations",
        "Design interpretability faithfulness metrics that validate attention-highlighted sensor channels against known failure-mode ground truth",
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
            "name": "Anomaly Transformer",
            "url": "https://arxiv.org/abs/2110.02642",
            "description": "Transformer architecture using an association-discrepancy criterion to distinguish normal from anomalous time-series patterns, used as the primary anomaly detection model for sensor degradation signatures.",
            "verified": True,
        },
        {
            "name": "Informer",
            "url": "https://arxiv.org/abs/2012.07436",
            "description": "Efficient long-sequence transformer with ProbSparse self-attention, used for modeling long historical sensor sequences and forecasting degradation trends to support early-warning detection.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1 for anomaly detection",
        "Area under the precision-recall curve (AUPRC)",
        "Remaining Useful Life (RUL) prediction error",
        "False alarm rate",
    ],
    "implementation_steps": [
        "Preprocess multivariate sensor telemetry (temperature, pressure, vibration, RPM) into fixed-length, normalized windows",
        "Handle missing or irregularly sampled readings via interpolation or masking strategies appropriate to each sensor channel",
        "Train a transformer-based anomaly detector in an unsupervised or self-supervised regime using a reconstruction or association-discrepancy objective",
        "Benchmark against LSTM-autoencoder and threshold-based statistical baselines on the same preprocessed data",
        "Evaluate early-warning lead time and Remaining Useful Life prediction accuracy against known maintenance and failure logs",
        "Apply attention-based interpretability analysis and validate highlighted sensor channels against documented failure-mode signatures",
    ],
    "unverified_suggestions": ["NASA C-MAPSS Turbofan Engine Degradation dataset", "NASA Prognostics Data Repository", "AI4I 2020 Predictive Maintenance dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets the label-scarcity challenge central to real aviation maintenance data by prioritizing unsupervised/self-supervised training regimes",
        "Including Remaining Useful Life prediction alongside binary anomaly detection metrics gives a more operationally relevant evaluation than classification accuracy alone",
        "The interpretability component is tied to a concrete stakeholder need (maintenance engineer trust and regulatory context) rather than treated as an afterthought",
        "Benchmarking against both LSTM-autoencoder and threshold-based baselines provides a reasonable spread of comparison points reflecting current and legacy practice",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how simulated C-MAPSS-style degradation patterns will be validated as representative of real in-service failure modes before claiming operational relevance",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host aviation-specific sensor telemetry, so the project depends heavily on the unverified NASA and AI4I dataset suggestions, and real fleet data access remains an open feasibility question",
        "The plan does not address the safety-critical false-negative risk explicitly: an anomaly detector missing a genuine degradation event carries materially different consequences than a false alarm, and the evaluation plan does not weight these asymmetrically",
    ],
    "missing_components": [
        "An explicit cost-sensitive evaluation framework that weights false negatives (missed degradation) more heavily than false positives, reflecting real safety consequences",
        "A plan for engaging domain experts (maintenance engineers, aviation safety regulators) to validate both interpretability outputs and failure-mode ground truth",
    ],
    "novelty_score": 0.66,
    "quality_score": 0.8,
}

CONFIDENCE_JSON = {"internal_consistency": 0.86, "specificity": 0.83, "completeness": 0.89}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based anomaly detection for predictive aviation maintenance, based on the retrieved abstract." for t in titles]
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
