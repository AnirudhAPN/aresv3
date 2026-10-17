import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

# The repo's .env sets LLM_PROVIDER=local, which makes every agent bypass
# llm_service.generate/judge entirely and call services.local_model directly
# (a real local LoRA model that needs a HuggingFace download). We only want
# to mock the LLM calls, so force the "api" code path before config loads,
# ensuring agents route through llm_service where our mocks are installed.
os.environ["LLM_PROVIDER"] = "api"

TOPIC = "Explainable Boosting Machines for Interpretable Clinical Risk Prediction"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate Explainable Boosting Machines (EBMs) as a glass-box alternative to black-box gradient boosting models for clinical risk prediction tasks.",
        "Quantify the trade-off between predictive performance and interpretability when using EBMs versus XGBoost, LightGBM, and deep learning baselines on clinical outcome data.",
        "Assess whether EBM shape functions and pairwise interaction terms produce clinically plausible risk factor relationships that clinicians can validate and trust.",
        "Investigate how EBM-derived feature attributions can be integrated into clinical decision support workflows without degrading downstream decision quality.",
    ],
    "research_questions": [
        "How does the predictive performance (AUROC, AUPRC) of EBMs compare to XGBoost, LightGBM, and deep neural networks on clinical risk prediction tasks such as ICU mortality or 30-day readmission?",
        "Do the shape functions learned by EBMs align with established clinical risk factor relationships, and can clinicians correctly interpret them?",
        "What is the computational and calibration cost of enforcing additive, interpretable structure in EBMs compared to unrestricted boosted ensembles?",
        "How do clinicians' trust and decision accuracy change when presented with EBM explanations versus post-hoc explanations (e.g., SHAP) of black-box models?",
    ],
    "keywords": [
        "explainable boosting machine",
        "interpretable machine learning",
        "clinical risk prediction",
        "generalized additive models",
        "glass-box models healthcare",
    ],
    "research_scope": (
        "This research focuses on the application of Explainable Boosting Machines (EBMs), a form of "
        "Generalized Additive Model with pairwise interactions (GA2M), to clinical risk prediction problems "
        "such as ICU mortality, sepsis onset, and hospital readmission. The scope includes comparative "
        "benchmarking against both black-box ensemble methods (XGBoost, LightGBM, random forests) and deep "
        "learning approaches, evaluation of interpretability through shape function analysis and clinician "
        "user studies, and practical considerations for deploying inherently interpretable models in "
        "regulated clinical environments where auditability and trust are paramount. Studies using purely "
        "synthetic data without clinical validation, and non-tabular modalities such as imaging or free-text "
        "notes, are considered out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of glass-box models like EBMs in healthcare as regulatory bodies push for auditable AI in clinical decision support.",
        "Widespread use of post-hoc explainability tools (SHAP, LIME) applied to black-box gradient boosting models as a substitute for inherent interpretability.",
        "Increasing benchmarking of EBMs against XGBoost and deep learning on public clinical datasets such as MIMIC-III/IV, showing near-parity in predictive performance.",
        "Emerging interest in pairwise interaction terms within GA2M models to capture clinically meaningful feature interactions without sacrificing transparency.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: large-scale prospective clinical trials evaluating whether EBM-based risk scores change clinician behavior or patient outcomes compared to standard-of-care scoring systems.",
            "based_on_papers": ["Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of EBM calibration and fairness across demographic subgroups in clinical risk prediction, particularly for underrepresented patient populations.",
            "based_on_papers": ["InterpretML: A Unified Framework for Machine Learning Interpretability"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: formal clinician user studies measuring whether EBM shape-function explanations improve diagnostic decision accuracy compared to standard black-box-plus-SHAP explanations.",
            "based_on_papers": ["A Unified Approach to Interpreting Model Predictions"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Conduct prospective, multi-site clinical trials that measure the real-world impact of EBM-based decision support on clinician workflow and patient outcomes.",
        "Develop standardized fairness and calibration audits for EBMs across demographic and comorbidity subgroups to support equitable deployment.",
        "Design controlled clinician user studies comparing EBM shape-function visualizations against post-hoc explanation methods for diagnostic trust and accuracy.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "MIMIC-III / MIMIC-IV",
            "url": "https://physionet.org/content/mimiciv/",
            "description": "De-identified ICU clinical records, a common healthcare-AI benchmark.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "XGBoost",
            "url": "https://xgboost.readthedocs.io/",
            "description": "Gradient-boosted decision tree ensemble used as a high-performance black-box baseline against which EBM predictive accuracy and calibration are compared.",
            "verified": True,
        },
        {
            "name": "LightGBM",
            "url": "https://lightgbm.readthedocs.io/",
            "description": "Histogram-based gradient boosting framework used as a second black-box baseline, chosen for its speed and strong performance on tabular clinical data.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "AUROC (area under the receiver operating characteristic curve)",
        "AUPRC (area under the precision-recall curve)",
        "Expected Calibration Error (ECE)",
        "Brier score",
    ],
    "implementation_steps": [
        "Extract and preprocess clinical features (vitals, labs, demographics, comorbidities) from MIMIC-III/IV, handling missingness with clinically informed imputation",
        "Define the prediction target (e.g., in-hospital mortality, 30-day readmission) and construct time-respecting train/validation/test splits to avoid leakage",
        "Train an EBM using InterpretML alongside XGBoost and LightGBM baselines with matched hyperparameter tuning budgets",
        "Evaluate predictive performance and calibration across all models using AUROC, AUPRC, ECE, and Brier score on the held-out test set",
        "Extract EBM shape functions and pairwise interaction terms and have clinical domain experts assess plausibility against known risk factor relationships",
        "Conduct subgroup analysis of calibration and discrimination across demographic strata to surface fairness concerns before any deployment recommendation",
    ],
    "unverified_suggestions": [
        "Integrate EBM outputs into a live clinical decision support dashboard piloted with practicing clinicians to gather qualitative trust feedback.",
        "Explore differentially private training of EBMs to enable multi-institution federated benchmarking without sharing raw patient data.",
        "Combine EBM shape functions with SHAP explanations of a black-box model to study agreement/disagreement in attributed risk factors.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped comparison between an inherently interpretable model class (EBMs) and established black-box baselines on a well-known clinical benchmark.",
        "Methodology specifies both predictive performance metrics and calibration metrics, which is appropriate for risk prediction in a clinical setting.",
        "Plan includes clinician-facing validation of shape functions, connecting model interpretability to actual clinical utility rather than treating it as an abstract property.",
        "Research questions are concrete and testable, with a clear path from data extraction through subgroup fairness analysis.",
    ],
    "weaknesses": [
        "The proposal relies primarily on MIMIC-III/IV, a single-institution ICU dataset, which limits generalizability claims to broader clinical risk prediction settings.",
        "No power analysis or sample size justification is provided for the planned clinician user study on shape-function interpretability.",
        "The evidence base cited is not sufficient to strongly support claims about EBMs improving clinical decision-making, since no prospective outcome studies were found in the reviewed literature.",
        "The proposal does not address how missing data mechanisms in EHR data (e.g., informative missingness) might bias both the EBM shape functions and the black-box baselines differently.",
    ],
    "missing_components": [
        "A pre-registered protocol or statistical analysis plan for the clinician user study comparing EBM and black-box explanations.",
        "An explicit fairness/equity evaluation plan detailing which demographic subgroups will be analyzed and what disparity thresholds would be considered actionable.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to interpretable machine learning for clinical risk prediction, based on the retrieved abstract." for t in titles]
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
