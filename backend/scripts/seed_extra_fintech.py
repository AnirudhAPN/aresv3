import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Attention-Based Neural Networks for Credit Risk Scoring in Fintech"

PLANNER_JSON = {
    "research_objectives": [
        "Design an attention-based neural network architecture for predicting borrower default risk from structured and behavioral fintech data.",
        "Evaluate the predictive performance of attention mechanisms against traditional credit scoring models such as logistic regression and gradient boosting.",
        "Investigate the interpretability of attention weights as a means of explaining individual credit decisions to regulators and applicants.",
        "Assess the robustness and fairness of attention-based credit risk models across demographic and socioeconomic subgroups.",
    ],
    "research_questions": [
        "Can self-attention mechanisms improve default prediction accuracy over gradient boosted trees on tabular fintech credit data?",
        "Do attention weights provide faithful and human-interpretable explanations of individual credit risk decisions?",
        "How sensitive are attention-based credit scoring models to class imbalance and limited historical default data?",
        "What is the impact of attention-based architectures on fairness metrics across protected demographic groups compared to baseline models?",
    ],
    "keywords": [
        "attention mechanism credit scoring",
        "transformer tabular data",
        "credit risk deep learning",
        "explainable AI fintech",
        "default prediction neural network",
    ],
    "research_scope": (
        "This research focuses on supervised binary classification of consumer and small-business credit default "
        "using attention-based neural network architectures applied to structured tabular and time-series transaction "
        "data typical of fintech lending platforms. It covers model architecture design, comparative benchmarking "
        "against tree-based and linear baselines, interpretability analysis via attention weights and post-hoc "
        "explanation methods, and fairness evaluation across demographic subgroups. The scope excludes fraud "
        "detection, anti-money-laundering applications, unstructured text or image-based alternative credit data "
        "(e.g., social media scraping), and real-time production deployment or infrastructure engineering concerns."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of transformer and self-attention architectures for tabular financial data, moving beyond their original NLP and vision domains.",
        "Increasing regulatory pressure for explainable credit decisions is driving interest in attention weights as an interpretability mechanism.",
        "Hybrid models combining gradient boosting with learned embeddings or attention layers are emerging as a compromise between accuracy and interpretability.",
        "Fairness-aware training objectives are increasingly incorporated into credit scoring pipelines to mitigate disparate impact across demographic groups.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: rigorous evaluation of whether attention weights in tabular credit models are faithful explanations rather than plausible-looking artifacts, especially compared against SHAP or LIME baselines.",
            "based_on_papers": ["Attention Is Not Explanation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of attention-based credit models under severe class imbalance typical of real-world default rates (often below 5%), including calibration of predicted probabilities.",
            "based_on_papers": ["TabTransformer: Tabular Data Modeling Using Contextual Embeddings"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-institutional fairness benchmarking of attention-based scoring models using standardized demographic parity and equalized odds metrics on realistic fintech lending datasets.",
            "based_on_papers": ["FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop hybrid architectures that combine attention-derived embeddings with gradient-boosted decision trees to balance accuracy and auditability.",
        "Create standardized fairness and explainability benchmarks specific to attention-based tabular credit scoring models.",
        "Investigate lightweight attention variants suitable for low-latency, on-device credit decisioning in resource-constrained fintech environments.",
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
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention based architecture adapted to tabular feature columns, enabling the model to learn pairwise interactions between borrower attributes without manual feature engineering.",
            "verified": True,
        },
        {
            "name": "XGBoost",
            "url": "https://arxiv.org/abs/1603.02754",
            "description": "Gradient boosted decision tree ensemble used as a strong, widely deployed baseline for tabular credit risk scoring against which attention-based models are benchmarked.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["AUC-ROC", "Kolmogorov-Smirnov statistic", "F1 score", "Expected Calibration Error"],
    "implementation_steps": [
        "Collect and preprocess structured borrower application and transaction history data, including handling missing values and categorical encoding",
        "Engineer time-windowed behavioral features such as payment history trends and utilization ratios",
        "Implement a tabular attention-based architecture with column embeddings and multi-head self-attention layers",
        "Train baseline logistic regression and XGBoost models on the same feature set for comparison",
        "Evaluate all models on held-out data using discrimination, calibration, and fairness metrics across demographic subgroups",
        "Analyze attention weight distributions and compare them against SHAP values to assess explanation faithfulness",
    ],
    "unverified_suggestions": ["Lending Club Loan Dataset", "Home Credit Default Risk Kaggle Dataset", "PyTorch Tabular library"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal targets a practically important problem with clear regulatory and business relevance in consumer lending.",
        "The comparison against both linear and gradient-boosted baselines provides a rigorous performance benchmark.",
        "Incorporating explainability analysis directly addresses a key adoption barrier for deep learning in regulated credit decisioning.",
        "The inclusion of fairness evaluation across demographic subgroups reflects awareness of real-world deployment risks.",
    ],
    "weaknesses": [
        "The proposal does not specify how class imbalance in default labels will be handled during training, which could bias reported performance.",
        "There is limited discussion of data privacy and regulatory compliance constraints (e.g., GDPR, ECOA) specific to consumer credit data.",
        "The evidence base drawn from the retrieved papers is relatively narrow, so claims about attention outperforming tree-based baselines on tabular fintech data should be treated as preliminary given the dataset fit has not been empirically confirmed.",
        "The proposal does not address computational cost or latency trade-offs of attention architectures relative to simpler baselines in production settings.",
    ],
    "missing_components": [
        "A concrete plan for obtaining or simulating a realistic, labeled fintech default dataset with sufficient positive examples.",
        "An ablation study design to isolate the contribution of the attention mechanism versus other architectural choices.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to attention-based credit risk scoring in fintech, based on the retrieved abstract." for t in titles]
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
