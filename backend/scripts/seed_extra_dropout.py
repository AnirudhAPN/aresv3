import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Networks for Predicting Student Dropout Risk in Online Learning Platforms"

PLANNER_JSON = {
    "research_objectives": [
        "Develop neural network models that predict individual student dropout risk in online learning platforms using clickstream, engagement, and assessment sequence data",
        "Compare recurrent and attention-based architectures for modeling temporal patterns in student interaction logs to identify early warning signals of disengagement",
        "Investigate which behavioral features (login frequency, video completion rate, forum participation, quiz latency) contribute most to accurate at-risk predictions",
        "Assess the trade-off between prediction lead time (how early a dropout can be flagged) and prediction accuracy for practical early-intervention use in online courses",
    ],
    "research_questions": [
        "Do sequence models (LSTM, Transformer) that encode the temporal order of student interactions outperform static feature-based classifiers for dropout prediction?",
        "Which behavioral and engagement features are most predictive of dropout risk across different online course formats (MOOCs versus for-credit online courses)?",
        "How does prediction accuracy degrade as the prediction window is moved earlier in the course, and what is an acceptable lead-time/accuracy trade-off for intervention?",
        "Can models trained on one course or platform generalize to a different course or platform without substantial performance loss?",
    ],
    "keywords": [
        "student dropout prediction",
        "learning analytics",
        "MOOC engagement modeling",
        "clickstream sequence classification",
        "at-risk student early warning system",
    ],
    "research_scope": (
        "This research covers neural network approaches for predicting student dropout or disengagement in online "
        "learning platforms, using behavioral log data such as clickstream events, video interactions, forum activity, "
        "and assessment performance as model inputs. It includes recurrent and attention-based sequence models and "
        "their comparison against traditional static classifiers. It excludes offline/in-person dropout prediction "
        "that does not rely on digital interaction logs, and it excludes curriculum design or pedagogical intervention "
        "content unrelated to the predictive modeling task itself."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of recurrent neural networks (LSTM, GRU) to model the temporal ordering of student clickstream events rather than aggregating them into static summary features",
        "Growing interest in attention-based and Transformer architectures for capturing long-range dependencies across a full course's interaction history",
        "Adoption of multi-modal feature fusion combining engagement logs, forum text, and assessment scores into a single predictive pipeline",
        "Emerging focus on early-prediction settings that evaluate models using only the first weeks of course data rather than the full course history",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of cross-course and cross-platform generalization of dropout prediction models trained on a single MOOC or LMS dataset.",
            "based_on_papers": ["Feature Engineering and Classifier Ensemble for KDD Cup 2015"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of the accuracy/lead-time trade-off across a common set of models and course datasets, since most studies fix the prediction window rather than sweeping it.",
            "based_on_papers": ["Early Prediction of Student Dropout in MOOCs using Deep Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: fairness analysis of dropout predictions across demographic subgroups, which matters for equitable deployment of early-intervention systems.",
            "based_on_papers": ["Feature Engineering and Classifier Ensemble for KDD Cup 2015"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build cross-platform benchmark suites that test dropout models trained on one course against held-out courses and platforms",
        "Establish standardized accuracy-versus-lead-time reporting so intervention timing trade-offs can be compared across studies",
        "Incorporate demographic fairness audits into dropout-prediction evaluation protocols before real-world deployment",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "LSTM (Long Short-Term Memory)",
            "url": "https://arxiv.org/abs/1503.04069",
            "description": "Recurrent architecture used to model the temporal sequence of student clickstream and engagement events for dropout prediction.",
            "verified": True,
        },
        {
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Attention-based architecture evaluated as an alternative to recurrent models for capturing long-range dependencies across a student's full interaction history.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "AUC-ROC for dropout classification",
        "F1-score at fixed prediction lead time",
        "Precision-recall curve across prediction windows",
        "Cross-course generalization accuracy drop",
    ],
    "implementation_steps": [
        "Collect and preprocess student interaction logs (clickstream, video events, forum posts, assessment scores) into per-student time-ordered sequences",
        "Label each student sequence with a dropout/completion outcome and define multiple prediction-window cutoffs (e.g., week 2, 4, 6)",
        "Implement LSTM and Transformer sequence classifiers alongside a static feature-based baseline (logistic regression or gradient boosting)",
        "Train and tune all models on a held-out validation split, sweeping the prediction-window cutoff to measure the accuracy/lead-time trade-off",
        "Evaluate cross-course and cross-platform generalization by training on one course and testing on a disjoint course or platform",
        "Conduct a subgroup analysis of prediction performance across available demographic or enrollment-type variables to check for disparate error rates",
    ],
    "unverified_suggestions": ["OULAD (Open University Learning Analytics Dataset)", "KDD Cup 2015 MOOC dropout dataset", "UCI Predict Students' Dropout and Academic Success dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly compares recurrent and attention-based sequence models against static baselines, giving a clear methodological contribution beyond a single-architecture study",
        "Explicitly sweeping the prediction-window cutoff to quantify the accuracy/lead-time trade-off addresses a practically important but often under-reported dimension",
        "The inclusion of cross-course and cross-platform generalization testing goes beyond the common single-dataset evaluation setup in this literature",
        "Adding a demographic subgroup fairness check reflects awareness of equity concerns relevant to real-world early-intervention deployment",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) is not a dedicated learning-analytics source, so the actual MOOC/LMS clickstream datasets remain unverified suggestions requiring separate access agreements.",
        "The proposal does not specify how missing or irregular logging (e.g., students who never engage with certain platform features) will be handled in the sequence representation.",
        "There is limited discussion of how model predictions would be translated into an actual intervention workflow, including false-positive costs of flagging students who would not have dropped out.",
    ],
    "missing_components": [
        "A concrete plan for handling missing or sparse interaction data within the sequence models.",
        "An analysis of intervention costs and false-positive trade-offs when deploying the model in a real advising workflow.",
    ],
    "novelty_score": 0.62,
    "quality_score": 0.72,
}

CONFIDENCE_JSON = {"internal_consistency": 0.78, "specificity": 0.76, "completeness": 0.80}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to predicting student dropout risk from online learning interaction data, based on the retrieved abstract." for t in titles]
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
