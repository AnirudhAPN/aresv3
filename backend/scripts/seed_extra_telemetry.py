import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Satellite Telemetry Anomaly Detection Using Transformer Architectures"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based anomaly detection models that operate directly on multivariate satellite housekeeping telemetry streams (temperature, voltage, current, attitude sensor readings)",
        "Evaluate self-attention architectures against LSTM-based and nonparametric dynamic thresholding baselines for detecting subsystem-level telemetry anomalies",
        "Investigate unsupervised and semi-supervised training regimes given the scarcity of confirmed anomaly labels in operational satellite telemetry archives",
        "Assess how attention weight patterns can be used to localize which telemetry channels and time windows drove a flagged anomaly, to support ground operator triage",
    ],
    "research_questions": [
        "Can transformer self-attention architectures detect satellite telemetry anomalies earlier or more reliably than LSTM-based or nonparametric dynamic thresholding methods?",
        "How should telemetry channel heterogeneity (differing sampling rates, units, and noise characteristics across subsystems) be handled in the input representation for a transformer anomaly detector?",
        "What unsupervised or self-supervised training objectives are most effective given that confirmed anomaly labels are rare in real satellite telemetry archives?",
        "How faithfully do attention-weight visualizations correspond to the telemetry channels that ground operators would identify as the true anomaly source?",
    ],
    "keywords": [
        "satellite telemetry anomaly detection",
        "transformer time series anomaly detection",
        "spacecraft housekeeping data",
        "self-attention multivariate time series",
        "nonparametric dynamic thresholding",
    ],
    "research_scope": (
        "This research focuses on transformer-based anomaly detection applied to multivariate satellite housekeeping "
        "telemetry, including power, thermal, and attitude sensor channels, covering unsupervised and semi-supervised "
        "training regimes and comparison against LSTM-based and nonparametric dynamic thresholding baselines. The "
        "scope includes input representation design for heterogeneous telemetry channels and attention-based "
        "localization of anomaly sources. It excludes onboard fault-tolerant control system design, physical circuit-level "
        "fault diagnosis, and ground-segment network or communications-link anomaly detection, which are treated as "
        "out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of transformer and self-attention architectures for multivariate time-series anomaly detection, extending beyond LSTM-based approaches such as Telemanom",
        "Continued reliance on nonparametric dynamic thresholding for flagging anomalies in reconstruction or prediction error, originally developed for spacecraft telemetry",
        "Increasing interest in association-discrepancy and reconstruction-based transformer objectives for detecting subtle, low-amplitude anomalies in sensor time series",
        "Limited public availability of labeled real spacecraft anomaly telemetry, driving reliance on a small number of open archives such as the SMAP and MSL telemetry datasets",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of transformer anomaly detectors across telemetry channels with heterogeneous sampling rates and units within the same spacecraft, rather than treating all channels as uniformly resampled inputs.",
            "based_on_papers": ["Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of transformer-based detectors against the nonparametric dynamic thresholding approach used in the widely cited Telemanom spacecraft anomaly detection system, on the same telemetry archive.",
            "based_on_papers": ["Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative validation of attention-weight-based anomaly localization against ground-truth root-cause telemetry channels identified by spacecraft operators, rather than qualitative attention visualizations alone.",
            "based_on_papers": ["Attention Is All You Need"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design channel-aware embedding schemes that explicitly encode sampling rate, unit, and subsystem identity for heterogeneous telemetry inputs to transformer models",
        "Establish shared benchmarks that directly compare transformer, LSTM, and dynamic-thresholding anomaly detectors on the same operational telemetry archives",
        "Develop interpretability faithfulness metrics that validate attention-based anomaly localization against operator-confirmed root-cause channels",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Transformer (Self-Attention Architecture)",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention-based sequence architecture adapted to encode multivariate telemetry windows, forming the core anomaly detection backbone in place of recurrent encoders.",
            "verified": True,
        },
        {
            "name": "Anomaly Transformer",
            "url": "https://arxiv.org/abs/2110.02642",
            "description": "Transformer variant using an association-discrepancy criterion between prior and series attention to distinguish normal telemetry patterns from anomalous ones without requiring labeled anomalies.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1-score for anomaly detection",
        "Area under the precision-recall curve (AUPRC)",
        "Detection lead time relative to confirmed anomaly onset",
        "False alarm rate per telemetry channel per day",
    ],
    "implementation_steps": [
        "Assemble multivariate telemetry windows from spacecraft housekeeping archives, aligning channels with differing sampling rates via resampling and channel-identity embeddings",
        "Normalize each telemetry channel independently to account for differing units and dynamic ranges across power, thermal, and attitude subsystems",
        "Train a transformer-based anomaly detector using a reconstruction or association-discrepancy objective on unlabeled or weakly labeled telemetry sequences",
        "Benchmark against an LSTM-based encoder and the nonparametric dynamic thresholding approach on the same held-out telemetry sequences",
        "Evaluate detection lead time and false alarm rate against operator-confirmed anomaly logs where available",
        "Analyze attention weights on confirmed anomaly cases and compare highlighted channels against operator-identified root-cause telemetry sources",
    ],
    "unverified_suggestions": [
        "NASA SMAP and MSL spacecraft telemetry anomaly dataset used in the Telemanom study",
        "ESA Anomaly Detection Benchmark (ESA-ADB) spacecraft telemetry dataset",
        "NASA Prognostics Data Repository",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds its evaluation plan in a well-known prior system (Telemanom's nonparametric dynamic thresholding), giving a concrete and credible baseline rather than only comparing against generic recurrent models",
        "Explicitly addressing telemetry channel heterogeneity (sampling rate, units, subsystem identity) shows attention to a real practical obstacle rather than assuming clean, uniform sensor input",
        "Including detection lead time and per-channel false alarm rate as metrics reflects operationally meaningful evaluation criteria for spacecraft operators, not just aggregate classification accuracy",
        "The plan to validate attention-based localization against operator-confirmed root causes goes beyond qualitative visualization, directly testing interpretability faithfulness",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how confirmed anomaly labels will be obtained for evaluation, given that the research questions themselves note label scarcity in operational archives",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host spacecraft telemetry archives, so the project depends on the unverified SMAP/MSL and ESA-ADB dataset suggestions being accessible and sufficiently representative",
        "The plan does not address how the model would generalize to a different spacecraft bus or instrument suite than the one used for training, which is a common failure mode for telemetry-specific anomaly detectors",
    ],
    "missing_components": [
        "A discussion of how false alarms translate into operator workload and mission risk, to justify the chosen precision/recall operating point",
        "A plan for periodically retraining or recalibrating the model as telemetry statistics drift over a spacecraft's mission lifetime",
    ],
    "novelty_score": 0.62,
    "quality_score": 0.78,
}

CONFIDENCE_JSON = {"internal_consistency": 0.85, "specificity": 0.79, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based satellite telemetry anomaly detection, based on the retrieved abstract." for t in titles]
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
