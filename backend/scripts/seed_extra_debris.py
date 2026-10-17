import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Space Debris Trajectory Prediction and Collision Avoidance"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning models that forecast the short-term orbital trajectory evolution of tracked space debris from historical two-line element (TLE) and conjunction data message (CDM) sequences",
        "Evaluate sequence models against physics-based propagators (e.g., SGP4) for probability-of-collision estimation accuracy in close-approach conjunction events",
        "Investigate physics-informed neural network formulations that constrain trajectory predictions to satisfy orbital mechanics rather than learning purely data-driven dynamics",
        "Assess how prediction uncertainty should be communicated to support automated or human-in-the-loop collision avoidance maneuver planning",
    ],
    "research_questions": [
        "Can deep sequence models improve short-term collision probability estimation accuracy over physics-based orbital propagators such as SGP4 for tracked debris conjunction events?",
        "How can physics-informed constraints be incorporated into neural trajectory prediction models to prevent physically implausible orbital state predictions?",
        "What is the effect of TLE measurement noise and update latency on the reliability of learned trajectory prediction models compared to classical propagators?",
        "How should predictive uncertainty in debris trajectory forecasts be quantified and communicated to support timely, well-calibrated collision avoidance maneuver decisions?",
    ],
    "keywords": [
        "space debris trajectory prediction",
        "collision avoidance satellite",
        "conjunction assessment machine learning",
        "orbital mechanics deep learning",
        "two-line element TLE forecasting",
    ],
    "research_scope": (
        "This research focuses on deep learning methods for forecasting orbital state evolution and estimating "
        "collision probability for tracked space debris, using historical two-line element and conjunction data "
        "message sequences. The scope covers sequence-model and physics-informed neural trajectory prediction, "
        "comparison against classical propagators such as SGP4, and uncertainty quantification to support "
        "collision avoidance maneuver planning. It excludes physical debris-removal mission and hardware design, "
        "detailed modeling of debris fragmentation events, and ground-based radar or optical tracking sensor "
        "design, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing interest in machine learning for conjunction assessment following community benchmarking efforts such as the ESA Kelvins Collision Avoidance Challenge",
        "Continued reliance on classical analytical and numerical propagators (e.g., SGP4, high-precision numerical integrators) as the operational standard against which learned methods are compared",
        "Increasing exploration of physics-informed neural network formulations to combine data-driven learning with known orbital mechanics constraints",
        "Rising attention to uncertainty quantification in collision probability estimates, motivated by the operational cost of both missed conjunctions and unnecessary avoidance maneuvers",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of learned trajectory prediction models against SGP4-class propagators under realistic TLE update latency and measurement noise, rather than idealized clean orbital state inputs.",
            "based_on_papers": ["Spacecraft Collision Avoidance Challenge: Design and Results of a Machine Learning Competition"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of physics-informed neural network constraints specifically for orbital trajectory prediction, as opposed to general physics-informed learning applications in fluid dynamics or structural mechanics.",
            "based_on_papers": ["Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: calibration analysis of predicted collision probability distributions against actual close-approach outcomes at the operational decision thresholds used for maneuver planning.",
            "based_on_papers": ["Spacecraft Collision Avoidance Challenge: Design and Results of a Machine Learning Competition"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop hybrid models that combine physics-informed constraints with learned residual corrections to capture unmodeled perturbations such as atmospheric drag variability",
        "Build evaluation protocols that explicitly inject realistic TLE noise and update latency to stress-test learned trajectory predictors under operational conditions",
        "Design calibration-aware collision probability estimators and study their effect on maneuver decision thresholds compared to point-estimate approaches",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Long Short-Term Memory (LSTM)",
            "url": "https://www.bioinf.jku.at/publications/older/2604.pdf",
            "description": "Recurrent architecture used to model sequential orbital state evolution from historical TLE and conjunction data message sequences for short-term trajectory forecasting.",
            "verified": True,
        },
        {
            "name": "Physics-Informed Neural Network (PINN)",
            "url": "https://arxiv.org/abs/1711.10561",
            "description": "Neural network formulation that incorporates known orbital mechanics equations as soft constraints in the loss function, used to keep trajectory predictions physically plausible.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Root Mean Squared Error (RMSE) of predicted orbital state vectors",
        "Probability-of-collision estimation error against reference propagator outcomes",
        "Brier score for calibration of predicted collision probabilities",
        "Maneuver decision precision and recall at operational alert thresholds",
    ],
    "implementation_steps": [
        "Collect historical TLE sequences and conjunction data messages for tracked debris and active satellites from public catalogs",
        "Preprocess orbital state vectors into consistent reference frames and time-aligned sequences, injecting realistic update latency and measurement noise to reflect operational conditions",
        "Train an LSTM-based sequence model to forecast short-term orbital state evolution from historical TLE windows",
        "Incorporate physics-informed constraints derived from two-body and perturbation equations into the training loss to penalize physically implausible trajectory predictions",
        "Compute probability-of-collision estimates from predicted trajectories and compare against SGP4-propagated reference estimates on held-out close-approach events",
        "Evaluate calibration of predicted collision probabilities using Brier score and assess precision/recall of maneuver-triggering decisions at realistic alert thresholds",
    ],
    "unverified_suggestions": [
        "ESA Kelvins Collision Avoidance Challenge dataset",
        "Space-Track.org TLE catalog",
        "NASA Orbital Debris Program Office debris catalog",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal is grounded in a concrete, well-known community benchmark (ESA Kelvins Collision Avoidance Challenge), giving it a credible empirical anchor rather than a purely theoretical framing",
        "Combining physics-informed constraints with a learned sequence model is a sound approach to the known risk of purely data-driven trajectory models producing physically implausible predictions",
        "Injecting realistic TLE noise and update latency into the evaluation protocol reflects genuine attention to operational conditions rather than testing only on idealized clean data",
        "Including calibration analysis (Brier score) alongside point-estimate error metrics directly addresses the operational need for trustworthy probability estimates, not just accurate point predictions",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how ground-truth collision outcomes (as opposed to propagator-estimated probabilities) will be obtained for calibration validation, since actual collisions are extremely rare events",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host orbital TLE or conjunction data, so the project depends heavily on the unverified Space-Track.org and Kelvins dataset suggestions being accessible and sufficiently comprehensive",
        "The plan does not address a genuinely important consideration for this domain: atmospheric drag variability driven by solar activity is a major source of orbital prediction error that neither the LSTM nor the stated physics-informed constraints are described as explicitly modeling",
    ],
    "missing_components": [
        "A discussion of how solar-activity-driven atmospheric drag variability will be incorporated as an input feature or constraint, given its known impact on low-Earth-orbit trajectory accuracy",
        "A plan for validating the operational usefulness of maneuver-triggering decisions against actual conjunction assessment practice, e.g., comparison with real operator decisions where available",
    ],
    "novelty_score": 0.70,
    "quality_score": 0.79,
}

CONFIDENCE_JSON = {"internal_consistency": 0.87, "specificity": 0.82, "completeness": 0.88}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning for space debris trajectory prediction and collision avoidance, based on the retrieved abstract." for t in titles]
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
