import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Machine Learning for Spacecraft Subsystem Health Monitoring"

PLANNER_JSON = {
    "research_objectives": [
        "Develop machine learning models for fault classification and remaining useful life (RUL) estimation across spacecraft subsystems, including electrical power, thermal control, propulsion, and attitude determination and control (ADCS)",
        "Evaluate classical ensemble methods (e.g., random forests, gradient boosted trees) against deep sequence models for subsystem-level health state classification from multi-sensor housekeeping data",
        "Investigate sensor fusion strategies that combine readings across subsystems to detect cross-subsystem fault propagation that single-subsystem models would miss",
        "Assess interpretability and confidence calibration of subsystem health models to support mission operator trust in automated health assessments",
    ],
    "research_questions": [
        "How do ensemble tree-based methods compare to LSTM-based sequence models for classifying subsystem health states from multi-sensor spacecraft housekeeping data?",
        "Can cross-subsystem sensor fusion detect fault propagation chains (e.g., thermal anomalies affecting power subsystem performance) that subsystem-isolated models cannot?",
        "What feature engineering and representation choices best capture degradation trends specific to each subsystem type (power, thermal, propulsion, ADCS)?",
        "How well-calibrated are model confidence outputs for health state classification, and does calibration quality affect operator trust in automated alerts?",
    ],
    "keywords": [
        "spacecraft health monitoring",
        "prognostics and health management",
        "subsystem fault classification",
        "remaining useful life spacecraft",
        "sensor fusion satellite subsystems",
    ],
    "research_scope": (
        "This research focuses on machine learning approaches for prognostics and health management (PHM) of "
        "spacecraft subsystems, specifically electrical power systems, thermal control, propulsion, and attitude "
        "determination and control, using multi-sensor housekeeping data. The scope covers subsystem-level fault "
        "classification, remaining useful life estimation, cross-subsystem sensor fusion for fault propagation "
        "detection, and confidence calibration for operator-facing health alerts. It excludes raw telemetry-stream "
        "anomaly detection at the signal level using deep sequence architectures, ground segment or mission control "
        "software, and physical hardware redesign, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Continued use of the NASA Advanced Diagnostics and Prognostics Testbed (ADAPT) electrical power system dataset as a standard benchmark for spacecraft subsystem fault diagnosis research",
        "Growing interest in ensemble tree-based methods (random forests, gradient boosting) for spacecraft health classification due to their robustness on limited, imbalanced fault-labeled data",
        "Increasing use of LSTM-based and sequence models for subsystem degradation trend forecasting, extending beyond static classification approaches",
        "Emerging emphasis on multi-subsystem sensor fusion, motivated by documented cases of fault propagation across spacecraft subsystems",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of cross-subsystem sensor fusion architectures that explicitly model fault propagation between subsystems, as opposed to training separate classifiers per subsystem.",
            "based_on_papers": ["Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: confidence calibration analysis for subsystem health classifiers, i.e., whether predicted probabilities reliably reflect true fault likelihood in a way operators can act on.",
            "based_on_papers": ["XGBoost: A Scalable Tree Boosting System"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparative benchmarking of ensemble tree-based methods against deep sequence models on the same subsystem health dataset under realistic class imbalance between nominal and faulted states.",
            "based_on_papers": ["XGBoost: A Scalable Tree Boosting System"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop graph-based or hierarchical models that represent spacecraft subsystems as interconnected nodes to explicitly capture fault propagation pathways",
        "Design calibration-aware training objectives (e.g., temperature scaling, focal loss variants) tailored to imbalanced spacecraft fault datasets",
        "Build shared cross-subsystem benchmark suites that allow fair comparison between classical ensemble methods and deep sequence models under consistent evaluation protocols",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "XGBoost (Gradient Boosted Trees)",
            "url": "https://arxiv.org/abs/1603.02754",
            "description": "Scalable gradient boosting framework used as the primary ensemble classifier for subsystem health state classification from tabular multi-sensor features.",
            "verified": True,
        },
        {
            "name": "Long Short-Term Memory (LSTM)",
            "url": "https://www.bioinf.jku.at/publications/older/2604.pdf",
            "description": "Recurrent architecture used to model degradation trends over time for remaining useful life estimation within each spacecraft subsystem.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1-score for fault classification",
        "Balanced accuracy under class imbalance between nominal and faulted states",
        "Remaining Useful Life (RUL) prediction error",
        "Expected Calibration Error (ECE) for predicted fault probabilities",
    ],
    "implementation_steps": [
        "Aggregate multi-sensor housekeeping data per subsystem (power, thermal, propulsion, ADCS), engineering subsystem-specific degradation features (e.g., voltage droop rate, thermal gradient trends)",
        "Label historical windows with nominal/faulted health states using documented anomaly and fault logs where available",
        "Train gradient boosted tree classifiers per subsystem as a baseline, addressing class imbalance via resampling or class-weighted loss",
        "Train LSTM-based sequence models on the same feature windows to compare against the tree-based baseline for both classification and RUL regression",
        "Construct a cross-subsystem fusion model that incorporates features from multiple subsystems to test detection of propagated faults",
        "Evaluate classification metrics, RUL error, and calibration error, and analyze cases where fusion improves detection over subsystem-isolated models",
    ],
    "unverified_suggestions": [
        "NASA Advanced Diagnostics and Prognostics Testbed (ADAPT) electrical power system dataset",
        "NASA Prognostics Data Repository",
        "ESA Anomaly Detection Benchmark (ESA-ADB) spacecraft telemetry dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal addresses a practically important but often overlooked problem: fault propagation across subsystems, rather than assuming subsystems can be monitored in isolation",
        "Including calibration error (ECE) alongside classification metrics reflects a mature understanding that operator trust depends on well-calibrated confidence, not just raw accuracy",
        "Comparing classical ensemble methods against LSTM sequence models on the same data gives a fair, informative baseline comparison rather than assuming deep learning is automatically superior",
        "The implementation plan explicitly addresses class imbalance between nominal and faulted states, which is a realistic and common obstacle in spacecraft fault datasets",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how fault labels for the cross-subsystem fusion model will be validated as true propagation events versus coincidental co-occurring anomalies",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host spacecraft subsystem datasets, so the project depends on the unverified NASA ADAPT and ESA-ADB suggestions being available and covering multiple subsystem types",
        "The plan does not address a genuinely important methodological consideration for this domain: differing sampling rates and units across subsystems (e.g., thermal sensors sampled far less frequently than power bus voltage) complicate the feature alignment step and are not discussed",
    ],
    "missing_components": [
        "A plan for validating model-flagged fault propagation chains against documented historical anomaly reports or subject-matter-expert review",
        "An ablation isolating how much the cross-subsystem fusion model improves detection over the best single-subsystem baseline",
    ],
    "novelty_score": 0.65,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.82, "specificity": 0.78, "completeness": 0.83}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to machine learning for spacecraft subsystem health monitoring, based on the retrieved abstract." for t in titles]
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
