import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Federated Learning for Cross-Silo Collaborative Fraud Detection in Banking"

PLANNER_JSON = {
    "research_objectives": [
        "Design a cross-silo federated learning framework that enables multiple banks to jointly train a fraud detection model without sharing raw transaction data.",
        "Evaluate the detection performance of federated fraud models against centralized and single-institution baselines under realistic non-IID transaction distributions.",
        "Investigate privacy-preserving aggregation techniques, such as secure aggregation and differential privacy, that limit information leakage between participating banks.",
        "Assess the robustness of the federated fraud detection system against poisoning attacks and malicious or unreliable participating institutions.",
    ],
    "research_questions": [
        "Can a federated learning model trained across multiple banks achieve fraud detection performance comparable to a centrally trained model on pooled transaction data?",
        "How does non-IID heterogeneity in fraud patterns and transaction volume across banks affect convergence and detection accuracy of the global federated model?",
        "What combination of secure aggregation and differential privacy provides an acceptable trade-off between fraud detection accuracy and formal privacy guarantees?",
        "How resilient is the cross-silo federated fraud detection system to data poisoning or model update manipulation by a compromised or adversarial bank participant?",
    ],
    "keywords": [
        "federated learning fraud detection",
        "cross-silo collaborative banking",
        "secure aggregation privacy",
        "non-IID transaction data",
        "differential privacy financial ML",
    ],
    "research_scope": (
        "This research focuses on cross-silo federated learning, in which a small number of banking institutions act as "
        "participants that collaboratively train a shared fraud detection model coordinated by a central server, without "
        "exchanging raw transaction records. It covers federated optimization algorithms, privacy-preserving aggregation "
        "mechanisms (secure aggregation and differential privacy), robustness to non-IID data distributions and adversarial "
        "participants, and evaluation against centralized and locally trained baselines on transaction-level fraud labels. "
        "The scope excludes cross-device federated learning with massive numbers of consumer devices, anti-money-laundering "
        "network analysis, and regulatory or legal frameworks for inter-bank data sharing beyond their technical implications "
        "for system design."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing interest in federated learning as a mechanism for banks to collaborate on fraud detection while complying with data localization and privacy regulations.",
        "Growing use of secure aggregation protocols to prevent the central server from inspecting individual bank model updates during training.",
        "Adoption of differential privacy noise injection into federated updates to provide formal guarantees against membership inference attacks on sensitive financial records.",
        "Emerging attention to Byzantine-robust aggregation rules to defend federated fraud models against malicious or compromised participating institutions.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: empirical benchmarking of cross-silo federated fraud detection under the extreme class imbalance and highly skewed non-IID fraud rates observed across real banking institutions of different sizes.",
            "based_on_papers": ["Federated Learning: Challenges, Methods, and Future Directions"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of the utility-privacy trade-off when combining secure aggregation with differential privacy specifically for transaction-level fraud features rather than generic tabular or image benchmarks.",
            "based_on_papers": ["Practical Secure Aggregation for Federated Learning on User-Held Data"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of Byzantine-robust aggregation methods against realistic model-poisoning strategies tailored to fraud label manipulation in a small cross-silo cohort of banks.",
            "based_on_papers": ["Federated Learning: Challenges, Methods, and Future Directions", "Practical Secure Aggregation for Federated Learning on User-Held Data"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop fraud-specific non-IID benchmarks that reflect realistic differences in transaction volume, customer demographics, and fraud typologies across banks.",
        "Design adaptive privacy budgets that allocate differential privacy noise based on each institution's data sensitivity and contribution size.",
        "Investigate hybrid Byzantine-robust and privacy-preserving aggregation rules that remain effective in the small-cohort, high-stakes cross-silo banking setting.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "World Bank Open Data",
            "url": "https://data.worldbank.org/",
            "description": "Global development, economic, and finance indicators.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Federated Averaging (FedAvg)",
            "url": "https://arxiv.org/abs/1602.05629",
            "description": "Baseline federated optimization algorithm in which participating banks train locally on their own transaction data and periodically average model updates on a central server without exchanging raw records.",
            "verified": True,
        },
        {
            "name": "XGBoost",
            "url": "https://arxiv.org/abs/1603.02754",
            "description": "Gradient boosted decision tree ensemble trained on pooled or single-institution transaction data, used as a centralized and local baseline against which the federated fraud detection model is benchmarked.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["AUC-PR", "Recall at fixed false positive rate", "F1 score", "Communication rounds to convergence"],
    "implementation_steps": [
        "Partition transaction data across simulated bank silos to reflect realistic non-IID fraud rates and volume differences",
        "Implement local model training pipelines at each simulated bank using a shared model architecture and feature schema",
        "Integrate a federated averaging coordinator with secure aggregation to combine local updates without exposing individual bank gradients",
        "Add configurable differential privacy noise injection to local updates and measure its effect on detection performance",
        "Train centralized and single-institution baseline models on the same underlying data splits for comparison",
        "Evaluate all models on held-out transactions using fraud detection metrics and simulate poisoning attacks to test robustness",
    ],
    "unverified_suggestions": ["IEEE-CIS Fraud Detection Kaggle Dataset", "PaySim Synthetic Financial Dataset", "Flower federated learning framework"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal addresses a practically significant problem where banks want to collaborate on fraud detection but face strict data privacy and regulatory constraints.",
        "The inclusion of both secure aggregation and differential privacy provides a layered approach to formal privacy protection.",
        "Benchmarking against centralized and single-institution baselines gives a clear, interpretable measure of the value of collaboration.",
        "The robustness evaluation against poisoning attacks reflects realistic threat models for a multi-institution financial system.",
    ],
    "weaknesses": [
        "The proposal does not specify how ground-truth fraud labels, which are often delayed or incomplete in real banking systems, will be handled across federated rounds.",
        "There is limited discussion of the computational and communication overhead that secure aggregation protocols impose on smaller participating banks.",
        "The evidence base drawn from the retrieved papers is relatively narrow, so claims about federated performance matching centralized training should be treated as preliminary given that dataset fit for realistic cross-silo banking fraud data has not been empirically confirmed.",
        "The proposal does not address how the number and identity of participating banks might change over time and how this affects model stability.",
    ],
    "missing_components": [
        "A concrete plan for simulating or obtaining realistic multi-institution transaction data with non-IID fraud characteristics.",
        "An explicit privacy budget accounting methodology for the differential privacy component across training rounds.",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to federated learning for cross-silo collaborative fraud detection in banking, based on the retrieved abstract." for t in titles]
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
