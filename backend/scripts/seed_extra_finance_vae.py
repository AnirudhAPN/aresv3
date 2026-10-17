import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Variational Autoencoders for Anomaly Detection in Financial Transactions"

PLANNER_JSON = {
    "research_objectives": [
        "Design a variational autoencoder (VAE) architecture that learns a compact latent representation of normal financial transaction behavior.",
        "Use per-transaction reconstruction error and latent-space likelihood from the VAE to flag anomalous transactions indicative of fraud.",
        "Compare VAE-based anomaly scoring against established baselines (Isolation Forest, standard autoencoders) on class-imbalanced transaction data.",
        "Assess the practical trade-offs between detection accuracy, false-positive rate, and inference latency for deployment in a real-time transaction-screening pipeline.",
    ],
    "research_questions": [
        "Can a VAE trained only on non-fraudulent transactions reliably separate fraudulent transactions using reconstruction probability as an anomaly score?",
        "How does VAE-based anomaly detection perform relative to Isolation Forest and standard (non-variational) autoencoders on highly imbalanced financial datasets?",
        "What latent dimensionality and KL-divergence weighting (beta) best balance reconstruction fidelity against a well-regularized, anomaly-discriminative latent space?",
        "How sensitive is detection performance to transaction feature engineering, such as time-since-last-transaction and amount normalization?",
    ],
    "keywords": [
        "variational autoencoder",
        "anomaly detection",
        "financial fraud detection",
        "unsupervised deep learning",
        "imbalanced classification",
    ],
    "research_scope": (
        "This research focuses on unsupervised and semi-supervised anomaly detection in structured, "
        "tabular financial transaction data using variational autoencoders. The scope covers model "
        "architecture design, training on predominantly normal transaction data, and evaluation of "
        "reconstruction-error and latent-likelihood based anomaly scores against classical baselines. "
        "It excludes graph-based fraud networks, natural-language transaction narratives, and real-time "
        "streaming infrastructure concerns, focusing instead on offline model performance using publicly "
        "available, class-imbalanced credit card and payment transaction datasets."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of autoencoder and VAE variants for unsupervised fraud detection to address the scarcity of labeled fraudulent examples.",
        "Growing use of reconstruction error thresholds combined with latent-space density estimation to score transaction anomalies.",
        "Benchmarking anomaly detectors primarily on the Kaggle Credit Card Fraud dataset due to its realistic class imbalance and anonymized PCA features.",
        "Interest in hybrid approaches that combine VAEs with classical detectors like Isolation Forest to improve robustness across different fraud patterns.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of beta-VAE regularization strength against detection performance on financial transaction data specifically.",
            "based_on_papers": ["Auto-Encoding Variational Bayes"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of VAE-based anomaly detectors under concept drift, where normal transaction behavior evolves over time.",
            "based_on_papers": ["Isolation Forest"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: interpretability of the learned latent space to explain why a specific transaction was flagged as anomalous, which is important for fraud analyst review workflows.",
            "based_on_papers": ["Auto-Encoding Variational Bayes", "Isolation Forest"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Investigate conditional or semi-supervised VAE variants that incorporate the small number of confirmed fraud labels typically available in production systems.",
        "Explore temporal or sequence-aware VAE architectures that model a customer's transaction history rather than scoring transactions independently.",
        "Study latent-space interpretability techniques, such as attaching feature-attribution methods to the VAE encoder, to support analyst decision-making.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "Yahoo Finance Historical Data", "url": "https://finance.yahoo.com/", "description": "Historical market price data commonly used in finance research.", "verified": True}],
    "algorithms": [
        {
            "name": "Autoencoder",
            "url": "https://www.deeplearningbook.org/",
            "description": "A neural network trained to reconstruct its input through a compressed bottleneck layer; reconstruction error on unseen data serves as an unsupervised anomaly score.",
            "verified": True,
        },
        {
            "name": "Isolation Forest",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html",
            "description": "An ensemble tree-based method that isolates anomalies by randomly partitioning feature space, requiring fewer splits to isolate outliers than normal points.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Area Under the Precision-Recall Curve (AUPRC)",
        "Area Under the ROC Curve (AUC-ROC)",
        "F1-score at the operating threshold",
        "Precision@k for the top-k highest-anomaly-score transactions",
    ],
    "implementation_steps": [
        "Collect and preprocess a labeled financial transaction dataset, normalizing amount and time-based features and excluding fraud labels from the training split.",
        "Train a variational autoencoder on normal transactions only, tuning latent dimensionality and the KL-divergence weighting term (beta).",
        "Derive an anomaly score for each transaction from reconstruction probability or reconstruction error combined with latent-space likelihood.",
        "Train Isolation Forest and a standard (non-variational) autoencoder as baselines on the same preprocessed data.",
        "Evaluate all models on a held-out test set containing both normal and fraudulent transactions using AUPRC, AUC-ROC, F1-score, and Precision@k.",
        "Perform ablation experiments varying beta and latent dimensionality to characterize the accuracy/regularization trade-off.",
    ],
    "unverified_suggestions": [
        "Consider a conditional VAE that incorporates merchant category as an auxiliary input to improve anomaly scoring for category-specific spending patterns.",
        "Explore ensembling the VAE reconstruction score with Isolation Forest's anomaly score via a simple weighted average to reduce false positives.",
        "Investigate synthetic minority oversampling (SMOTE) applied only to the validation split to better calibrate the anomaly-score threshold.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly grounds VAE-based anomaly detection in a well-established generative modeling framework (Auto-Encoding Variational Bayes) with a direct application to financial fraud.",
        "The methodology includes strong, widely accepted classical baselines (Isolation Forest, standard autoencoder), enabling meaningful comparative evaluation.",
        "Evaluation metrics are appropriate for a highly class-imbalanced task, favoring AUPRC and Precision@k over accuracy.",
        "The implementation steps form a coherent, reproducible pipeline from preprocessing through ablation analysis.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small for a topic as heavily studied as fraud detection, so conclusions about the state of the art should be treated as preliminary.",
        "The proposal does not specify which financial transaction dataset will be used beyond the whitelist-verified Yahoo Finance market data, which is a price time-series and not a transaction-level fraud dataset, creating a potential dataset-fit mismatch.",
        "No discussion of how class imbalance during training itself (beyond evaluation) will be handled, e.g., whether any fraud examples leak into VAE training.",
        "Latent-space interpretability and analyst-facing explanations are identified as a gap but not incorporated into the proposed methodology.",
    ],
    "missing_components": [
        "A concrete, transaction-level labeled dataset (e.g., a Kaggle credit card fraud dataset) with justification for its use in place of or alongside the verified market-data source.",
        "A plan for hyperparameter search strategy (e.g., grid or Bayesian search over beta and latent dimensionality) rather than ad hoc ablation.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to VAE-based anomaly detection in financial transactions, based on the retrieved abstract." for t in titles]
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
