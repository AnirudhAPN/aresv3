import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Privacy-Preserving Deep Learning via Differential Privacy for Healthcare Records"

PLANNER_JSON = {
    "research_objectives": [
        "Design differentially private deep learning training procedures that provide formal (epsilon, delta) privacy guarantees for models trained on sensitive electronic health record (EHR) data.",
        "Quantify the trade-off between privacy budget (epsilon) and downstream clinical prediction accuracy for models trained with differentially private stochastic gradient descent (DP-SGD) on healthcare records.",
        "Evaluate whether differentially private training meaningfully reduces membership-inference and re-identification risk for patients represented in EHR training sets.",
        "Investigate architectural and optimization techniques (gradient clipping schedules, noise calibration, per-example gradient computation) that reduce the utility cost of differential privacy in clinical deep learning models.",
    ],
    "research_questions": [
        "How does varying the privacy budget epsilon affect diagnostic and outcome-prediction accuracy of deep learning models trained on de-identified EHR data with DP-SGD?",
        "What gradient clipping and noise-calibration strategies best preserve clinical model utility under a fixed differential privacy guarantee?",
        "To what extent does differentially private training reduce empirical membership-inference attack success rates compared to non-private baselines on the same healthcare dataset?",
        "How do differentially private healthcare models compare to federated learning and other privacy-enhancing approaches in terms of the privacy-utility trade-off for clinical prediction tasks?",
    ],
    "keywords": [
        "differential privacy deep learning",
        "DP-SGD healthcare records",
        "privacy-preserving clinical prediction",
        "membership inference EHR",
        "privacy budget epsilon clinical models",
    ],
    "research_scope": "This research focuses on applying differential privacy, particularly DP-SGD and its variants, to deep learning models trained on electronic health record data for clinical prediction tasks such as diagnosis, mortality, and readmission prediction. The scope covers privacy budget selection, gradient clipping and noise calibration strategies, and empirical evaluation of both predictive utility and resistance to membership-inference and re-identification attacks on de-identified ICU and hospital record datasets. It excludes non-deep-learning privacy techniques (e.g., k-anonymity, pure statistical disclosure control), privacy-preserving methods applied outside the healthcare domain, and cryptographic approaches such as secure multiparty computation or homomorphic encryption, which are treated as out of scope for this study.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of DP-SGD with per-example gradient clipping and Gaussian noise addition as the default mechanism for training differentially private neural networks on clinical data.",
        "Growing use of moments accountant and Renyi differential privacy accounting to obtain tighter (epsilon, delta) budget estimates over many training epochs.",
        "Increasing interest in combining differential privacy with federated learning so that healthcare records never leave the originating institution while still bounding leakage from shared model updates.",
        "Persistent reporting of substantial utility degradation on clinical prediction tasks at privacy budgets (epsilon less than 1) considered strong enough for regulatory comfort.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of how differentially private training affects performance disparities across patient subgroups (e.g., by age, race, or comorbidity) in clinical prediction models.",
            "based_on_papers": ["Deep Learning with Differential Privacy"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: empirical comparison of DP-SGD against membership-inference attacks specifically calibrated to the small, imbalanced patient cohorts typical of single-institution EHR datasets.",
            "based_on_papers": ["Membership Inference Attacks Against Machine Learning Models"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized guidance for selecting a clinically defensible privacy budget epsilon that regulators, clinicians, and patients can jointly interpret as an acceptable disclosure risk.",
            "based_on_papers": ["Deep Learning with Differential Privacy", "Membership Inference Attacks Against Machine Learning Models"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop subgroup-aware differentially private training objectives that explicitly bound utility loss disparities across patient demographic groups.",
        "Design adaptive privacy budget allocation schemes that spend epsilon more efficiently on the clinically most informative training epochs or model layers.",
        "Establish standardized, clinically grounded benchmarks for reporting the privacy-utility trade-off of DP deep learning models on healthcare prediction tasks.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "MIMIC-III / MIMIC-IV", "url": "https://physionet.org/content/mimiciv/", "description": "De-identified ICU clinical records, a common healthcare-AI benchmark.", "verified": True}],
    "algorithms": [
        {
            "name": "Federated Averaging (FedAvg)",
            "url": "https://arxiv.org/abs/1602.05629",
            "description": "Core federated-learning algorithm used as a complementary privacy-preserving baseline, aggregating locally trained updates from multiple clinical sites without centralizing patient records.",
            "verified": True,
        },
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://www.tensorflow.org/tutorials/images/cnn",
            "description": "Base architecture trained with DP-SGD (per-example gradient clipping plus calibrated Gaussian noise) on structured and time-series EHR features to produce differentially private clinical predictions.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["AUROC", "AUPRC", "Privacy budget (epsilon) at fixed delta", "Membership-inference attack success rate"],
    "implementation_steps": [
        "Preprocess and de-identify MIMIC-III/IV records, extracting structured features (vitals, labs, diagnoses) and constructing patient-level train/validation/test splits.",
        "Implement DP-SGD with per-example gradient clipping and Gaussian noise addition, using a moments accountant to track cumulative privacy loss across training epochs.",
        "Train baseline non-private models and a sweep of differentially private models across a range of epsilon values on the same clinical prediction task.",
        "Evaluate predictive utility (AUROC, AUPRC) for each epsilon setting and quantify the utility-privacy trade-off curve.",
        "Run membership-inference attacks against both private and non-private models to empirically measure the reduction in disclosure risk achieved by DP-SGD.",
        "Compare the DP-SGD results against a federated-averaging baseline trained across simulated multi-institution splits of the same cohort.",
    ],
    "unverified_suggestions": ["DP-SGD (Differentially Private SGD) as a standalone verified algorithm entry", "eICU Collaborative Research Database", "Opacus differential privacy training library"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal ties privacy budget selection directly to measurable clinical utility metrics, making the privacy-utility trade-off empirically testable rather than purely theoretical.",
        "Including a federated-averaging comparison alongside DP-SGD gives the study a meaningful point of contrast among privacy-preserving training strategies.",
        "Membership-inference evaluation is built into the methodology as a direct empirical test of the privacy guarantee's practical effect, not just its theoretical bound.",
        "The research scope is clearly bounded, explicitly excluding cryptographic privacy techniques to keep the study focused on differentially private deep learning.",
    ],
    "weaknesses": [
        "The evidence sufficiency for the identified gaps is limited, resting on only 8 retrieved papers rather than a comprehensive systematic review of the differential privacy literature.",
        "MIMIC-III/IV alone may not be sufficient to validate findings on subgroup utility disparities, since its patient population and single-institution origin limit generalizability to other healthcare settings.",
        "The methodology does not specify how the delta parameter will be fixed or justified alongside the epsilon sweep, which is necessary for a complete (epsilon, delta)-DP guarantee.",
        "No concrete plan is given for how the moments accountant's cumulative privacy loss will be validated against an independent DP accounting method.",
    ],
    "missing_components": [
        "A plan for auditing subgroup-level utility disparities introduced by differentially private training, as flagged in the gap analysis.",
        "A clinically grounded justification process for selecting the final deployed epsilon value, involving regulatory or ethical review criteria.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to differentially private deep learning for healthcare records, based on the retrieved abstract." for t in titles]
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
