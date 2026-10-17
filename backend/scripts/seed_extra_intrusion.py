import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Anomaly-Based Intrusion Detection Using Deep Autoencoders in Enterprise Networks"

PLANNER_JSON = {
    "research_objectives": [
        "Design deep autoencoder architectures that learn compact representations of normal enterprise network traffic to enable unsupervised detection of anomalous or malicious flows.",
        "Evaluate the trade-off between detection accuracy and false-positive rate when using reconstruction-error-based anomaly scoring on production-scale enterprise network traffic.",
        "Investigate how autoencoder-based intrusion detection generalizes across previously unseen attack types not present in the training distribution of normal traffic.",
        "Assess the operational feasibility of deploying deep autoencoder intrusion detectors in real-time enterprise network monitoring pipelines.",
    ],
    "research_questions": [
        "How effectively can deep autoencoders distinguish anomalous network flows from normal enterprise traffic using unsupervised reconstruction error alone?",
        "What autoencoder architectural choices (dense, convolutional, variational, LSTM-based) yield the best balance of detection accuracy and false-positive rate on enterprise intrusion datasets?",
        "How well do autoencoder-based intrusion detection models generalize to novel or zero-day attack types absent from training data?",
        "What are the computational and latency trade-offs of deploying autoencoder-based anomaly detectors for real-time enterprise network monitoring?",
    ],
    "keywords": [
        "deep autoencoder intrusion detection",
        "anomaly-based network intrusion detection",
        "unsupervised network anomaly detection",
        "enterprise network security deep learning",
        "reconstruction error anomaly scoring",
    ],
    "research_scope": "This research focuses on deep autoencoder-based approaches to anomaly detection for network intrusion detection systems (NIDS) operating within enterprise network environments. The scope covers unsupervised and semi-supervised autoencoder variants (vanilla, convolutional, variational, and LSTM-based autoencoders) trained on normal traffic flows and evaluated on labeled intrusion benchmark datasets, with emphasis on detection accuracy, false-positive rate, and generalization to unseen attack categories. It excludes signature-based and rule-based intrusion detection systems, host-based intrusion detection, and anomaly detection techniques applied outside network traffic (e.g., industrial control systems or IoT sensor telemetry), which fall outside the enterprise network NIDS focus of this study.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread use of vanilla and variational autoencoders trained on benign traffic to flag anomalies via reconstruction error thresholds.",
        "Growing adoption of hybrid architectures combining autoencoders with LSTM or attention layers to capture temporal patterns in network flow sequences.",
        "Increasing reliance on public benchmark datasets such as CICIDS2017 and NSL-KDD for training and evaluating deep anomaly-based intrusion detectors.",
        "Emerging interest in combining autoencoder reconstruction scores with ensemble or isolation-based methods to reduce false-positive rates.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of autoencoder-based detectors against adversarially crafted network flows designed to evade reconstruction-error thresholds.",
            "based_on_papers": ["A Deep Learning Approach to Network Intrusion Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized methodology for setting and adapting reconstruction-error thresholds as enterprise traffic patterns drift over time.",
            "based_on_papers": ["Anomaly Detection Using Autoencoders in Network Traffic"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: large-scale comparison of autoencoder variants under realistic enterprise-scale traffic volume and class imbalance rather than curated benchmark subsets.",
            "based_on_papers": ["A Deep Learning Approach to Network Intrusion Detection", "Anomaly Detection Using Autoencoders in Network Traffic"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop adaptive thresholding techniques that recalibrate reconstruction-error cutoffs online as enterprise network baseline traffic evolves.",
        "Study adversarial robustness of autoencoder-based intrusion detectors and design training regimes that resist evasion by crafted traffic perturbations.",
        "Create realistic, large-scale enterprise traffic benchmarks with diverse and evolving attack categories to better evaluate generalization to zero-day threats.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "CICIDS2017", "url": "https://www.unb.ca/cic/datasets/ids-2017.html", "description": "Realistic network intrusion detection dataset with labeled attack traffic.", "verified": True}],
    "algorithms": [
        {
            "name": "Autoencoder",
            "url": "https://arxiv.org/abs/2003.05991",
            "description": "Unsupervised neural network trained to reconstruct normal enterprise traffic flow features; flows with high reconstruction error are flagged as anomalous intrusions.",
            "verified": True,
        },
        {
            "name": "Isolation Forest",
            "url": "https://arxiv.org/abs/1811.02141",
            "description": "Tree-based unsupervised anomaly detector used as a complementary baseline to isolate anomalous network flows based on partitioning behavior rather than reconstruction error.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["Precision", "Recall", "F1-score", "False Positive Rate"],
    "implementation_steps": [
        "Collect and preprocess raw enterprise network flow data, including feature extraction, normalization, and encoding of categorical protocol fields.",
        "Split the dataset into training, validation, and test sets, ensuring the autoencoder is trained predominantly on benign traffic flows.",
        "Train baseline and variational autoencoder architectures on normal traffic to learn a compact latent representation of benign network behavior.",
        "Determine reconstruction-error anomaly thresholds using validation data, balancing detection sensitivity against false-positive rate.",
        "Evaluate trained models on held-out labeled test traffic containing both normal and attack flows, including unseen attack categories.",
        "Benchmark inference latency and throughput of the trained autoencoder to assess feasibility for real-time enterprise network monitoring.",
    ],
    "unverified_suggestions": ["NSL-KDD Dataset", "UNSW-NB15 Dataset", "CIC-DDoS2019 Dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal presents clearly measurable research objectives grounded in realistic enterprise network security constraints.",
        "The methodology pairs a reconstruction-based deep learning method with a complementary non-neural baseline, enabling meaningful comparative evaluation.",
        "Evaluation metrics appropriately emphasize false-positive rate alongside standard classification metrics, reflecting operational NIDS concerns.",
        "The research scope is well-bounded, explicitly excluding host-based and non-network anomaly detection domains to maintain focus.",
    ],
    "weaknesses": [
        "The evidence sufficiency for the identified gaps is limited, resting on only 8 retrieved papers rather than a comprehensive systematic review.",
        "The proposal does not clearly justify why CICIDS2017 alone is sufficient for validating generalization to novel enterprise traffic and attack patterns given known dataset fit concerns.",
        "Adversarial robustness is flagged as a gap but no concrete adversarial training or evaluation protocol is specified in the methodology.",
        "The plan lacks detail on how reconstruction-error thresholds will be validated across different enterprise network segments with heterogeneous baseline traffic.",
    ],
    "missing_components": [
        "A plan for handling concept drift and periodic retraining of the autoencoder as enterprise network baselines evolve.",
        "An ablation study isolating the contribution of the variational component versus the vanilla autoencoder architecture.",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep autoencoder-based anomaly detection for enterprise network intrusion, based on the retrieved abstract." for t in titles]
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
