import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer-Based Anomaly Detection in Industrial Control Systems"

PLANNER_JSON = {
    "research_objectives": [
        "Design a transformer-based sequence model that detects anomalous behavior in industrial control system (ICS) sensor and actuator telemetry",
        "Evaluate whether self-attention mechanisms can capture long-range temporal dependencies in ICS process variables better than recurrent baselines",
        "Assess the model's robustness to stealthy, gradually-drifting sensor attacks that evade threshold-based detectors",
        "Quantify the trade-off between detection latency and false-positive rate in a near-real-time ICS monitoring setting",
    ],
    "research_questions": [
        "Can transformer encoder architectures outperform LSTM and CNN baselines for detecting anomalies in multivariate ICS time-series data?",
        "How does attention-based modeling handle the mixed continuous/discrete signal types typical of SCADA and PLC telemetry?",
        "What is the detection performance against stealthy, low-magnitude sensor spoofing attacks compared to abrupt fault injections?",
        "How well does a model trained on one ICS testbed (e.g., water treatment) transfer to a structurally different testbed (e.g., power grid)?",
    ],
    "keywords": [
        "transformer anomaly detection",
        "industrial control systems",
        "SCADA intrusion detection",
        "cyber-physical systems security",
        "time-series self-attention",
    ],
    "research_scope": "This research focuses on applying transformer-based deep learning architectures to anomaly and intrusion detection in industrial control systems (ICS), including SCADA and PLC-driven environments such as water treatment, power distribution, and manufacturing testbeds. The scope covers model design for multivariate sensor/actuator time-series, comparison against recurrent and statistical baselines, and evaluation on both abrupt fault scenarios and stealthy adversarial sensor manipulation. It excludes physical-layer signal processing and focuses instead on data-driven detection using publicly available ICS telemetry datasets.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of self-attention and transformer encoders for multivariate time-series anomaly detection, moving beyond LSTM/GRU-based sequence models",
        "Increasing use of public ICS testbed datasets (SWaT, WADI, HAI) to benchmark intrusion detection methods under realistic physical process dynamics",
        "Shift toward unsupervised and semi-supervised reconstruction- or forecasting-error based anomaly scoring due to scarcity of labeled attack data",
        "Rising interest in adversarial robustness evaluation, since attackers can craft stealthy sensor manipulations that evade naive statistical thresholds",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of transformer attention patterns against gradually-drifting, low-magnitude stealthy attacks rather than only abrupt, high-magnitude faults.",
            "based_on_papers": ["A Comprehensive Survey of Deep Learning for Anomaly Detection in ICS Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-testbed transferability of a trained transformer model, e.g. from a water treatment plant to a power grid or manufacturing process, without full retraining.",
            "based_on_papers": ["Anomaly Detection in Cyber-Physical Systems using Recurrent Neural Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: interpretability of transformer attention weights as a diagnostic tool for ICS operators to localize the physical sensor or actuator responsible for a flagged anomaly.",
            "based_on_papers": ["Attention-Based Models for Multivariate Time Series Anomaly Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop hybrid architectures combining transformer temporal attention with graph-based modeling of physical process topology to improve stealthy-attack sensitivity",
        "Investigate few-shot or meta-learning approaches to transfer a pretrained ICS transformer detector across heterogeneous testbeds with minimal fine-tuning",
        "Build attention-weight visualization tooling that maps flagged anomalies back to specific sensors/actuators to support operator trust and root-cause analysis",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "CICIDS2017",
            "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "description": "Realistic network intrusion detection dataset with labeled attack traffic.",
            "verified": True,
        },
        {
            "name": "SWaT (Secure Water Treatment)",
            "url": "https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/",
            "description": "Physical testbed dataset from a scaled-down water treatment plant with labeled normal operation and 36 cyber-physical attack scenarios spanning sensor spoofing, actuator manipulation, and network-based attacks.",
            "verified": True,
        },
    ],
    "algorithms": [
        {
            "name": "Transformer Encoder (multi-head self-attention)",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Sequence encoder using multi-head self-attention to model long-range temporal dependencies across ICS process variables, used here as a forecasting/reconstruction backbone for anomaly scoring.",
            "verified": True,
        },
        {
            "name": "LSTM Autoencoder",
            "url": "https://arxiv.org/abs/1607.00148",
            "description": "Recurrent encoder-decoder baseline trained to reconstruct normal ICS telemetry sequences, with reconstruction error used as the anomaly score for comparison against the transformer model.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1-score on labeled attack windows",
        "Area Under the ROC Curve (AUC-ROC) across varying anomaly-score thresholds",
        "Detection latency, measured as time steps between attack onset and first alarm",
        "False Positive Rate during normal (attack-free) operating periods",
    ],
    "implementation_steps": [
        "Preprocess SWaT/WADI telemetry into fixed-length sliding windows with per-sensor normalization and timestamp alignment across PLC polling rates",
        "Train the transformer encoder in a self-supervised forecasting or masked-reconstruction setup using only attack-free training data",
        "Implement LSTM autoencoder and statistical (e.g., PCA-based) baselines under identical preprocessing and windowing for fair comparison",
        "Define an anomaly score from reconstruction/forecast error and calibrate detection thresholds via a held-out validation split",
        "Evaluate all models on labeled test attacks, separately reporting performance on abrupt versus stealthy/gradual attack subsets",
        "Analyze transformer attention weights on flagged windows to assess whether they localize to the sensors actually manipulated during each attack",
    ],
    "unverified_suggestions": [
        "Consider augmenting training data with synthetic stealthy attacks generated via gradual sensor drift injection, since public datasets under-represent this attack class",
        "A graph neural network layer encoding known physical process topology (pipe/valve connectivity) could be added as a future extension to the transformer backbone",
        "Federated training across multiple ICS testbeds might improve cross-domain generalization but was not evaluated in the retrieved literature",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clear articulation of the gap between abrupt-fault detection and stealthy, gradually-drifting attack detection in ICS anomaly detection literature",
        "Methodology grounds the transformer approach against a well-established recurrent baseline (LSTM autoencoder) rather than proposing the model in isolation",
        "Uses a widely-adopted, physically realistic public testbed (SWaT) that provides labeled attack scenarios rather than relying solely on synthetic data",
        "Evaluation plan explicitly separates abrupt versus stealthy attack performance, directly addressing one of the identified literature gaps",
    ],
    "weaknesses": [
        "The proposal relies on a single primary testbed (SWaT); evidence for cross-domain generalization claims would need additional datasets such as WADI or HAI to be sufficiently supported",
        "No explicit adversarial threat model is defined for what constitutes a 'stealthy' attack magnitude, which weakens reproducibility of the stealthy-attack evaluation",
        "Attention-weight interpretability is proposed as a future direction but lacks a concrete quantitative validation plan against ground-truth attacked sensors",
        "The plan does not address class imbalance between normal and attack windows, which can bias precision/recall estimates if not explicitly handled",
    ],
    "missing_components": [
        "A concrete hyperparameter search strategy (window length, number of attention heads/layers) with justification tied to ICS sampling rates",
        "A statistical significance test (e.g., paired comparison across multiple random seeds) to support claims that the transformer outperforms baselines",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based anomaly detection in industrial control systems, based on the retrieved abstract." for t in titles]
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
