import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Anomaly Detection in Aircraft Sensor Data Using Variational Autoencoders"

PLANNER_JSON = {
    "research_objectives": [
        "Develop variational autoencoder (VAE) models that learn a probabilistic latent representation of nominal multivariate aircraft flight data recorder (FDR) sensor sequences",
        "Evaluate reconstruction-probability-based anomaly scoring from VAEs against deterministic autoencoder and PCA-based reconstruction error baselines",
        "Investigate LSTM-VAE architectures that model temporal dependencies within flight sensor sequences rather than treating each timestep independently",
        "Assess the sensitivity of VAE-based anomaly scores to normal operational variability (different flight phases, aircraft types) versus genuine sensor or system anomalies",
    ],
    "research_questions": [
        "Do variational autoencoders, using probabilistic reconstruction likelihood as an anomaly score, detect aircraft sensor anomalies more reliably than deterministic autoencoders or PCA reconstruction error?",
        "How does incorporating temporal structure via an LSTM-VAE affect anomaly detection performance compared to a feedforward VAE applied to individual sensor snapshots?",
        "How well do VAE-based anomaly scores distinguish genuine sensor or subsystem anomalies from normal variability across different flight phases (takeoff, cruise, descent)?",
        "What is the effect of latent space dimensionality on the trade-off between reconstruction fidelity for nominal data and sensitivity to true anomalies?",
    ],
    "keywords": [
        "variational autoencoder anomaly detection",
        "aircraft flight data recorder sensors",
        "reconstruction probability anomaly scoring",
        "LSTM variational autoencoder time series",
        "unsupervised aviation sensor anomaly detection",
    ],
    "research_scope": (
        "This research focuses on variational autoencoder methods for unsupervised anomaly detection in "
        "multivariate aircraft sensor time series drawn from flight data recorder and quick access recorder "
        "parameters, covering probabilistic reconstruction-based anomaly scoring, latent space design, and "
        "LSTM-VAE extensions for temporal modeling. It benchmarks against deterministic autoencoder and PCA "
        "reconstruction baselines. The scope excludes transformer-based sequence anomaly detectors, airframe "
        "structural fatigue analysis, and air traffic control or flight scheduling systems, which are treated as "
        "out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of variational autoencoders for unsupervised anomaly detection in multivariate sensor time series across safety-critical domains, extending the original reconstruction-probability framework",
        "Increasing adoption of LSTM-VAE architectures to capture temporal dependencies that feedforward VAEs applied to individual timesteps cannot represent",
        "Continued reliance on simulated or semi-synthetic degradation benchmarks due to limited public access to real in-service aircraft sensor anomaly data",
        "Rising interest in probabilistic anomaly scores (reconstruction probability) over simple reconstruction error, motivated by better-calibrated detection thresholds",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of VAE-based anomaly detectors on real flight data recorder sequences spanning multiple flight phases, as opposed to steady-state or simulated sensor benchmarks.",
            "based_on_papers": ["Auto-Encoding Variational Bayes"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison between reconstruction-probability-based anomaly scoring and simple reconstruction-error thresholds under matched model capacity and latent dimensionality.",
            "based_on_papers": ["Variational Autoencoder based Anomaly Detection using Reconstruction Probability"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of how flight-phase variability (takeoff, cruise, descent) affects false positive rates in VAE-based anomaly detectors trained without explicit phase conditioning.",
            "based_on_papers": ["A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop flight-phase-conditioned VAE architectures that explicitly account for expected variability across takeoff, cruise, and descent regimes",
        "Establish evaluation protocols using anonymized real flight data recorder archives to validate findings beyond simulated degradation benchmarks",
        "Study hybrid scoring functions that combine reconstruction probability with latent-space density estimates to improve anomaly score calibration",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Variational Autoencoder (VAE)",
            "url": "https://arxiv.org/abs/1312.6114",
            "description": "Generative latent-variable model trained via the reparameterization trick, used to learn a probabilistic representation of nominal aircraft sensor patterns and score anomalies via reconstruction probability.",
            "verified": True,
        },
        {
            "name": "LSTM-VAE",
            "url": "https://arxiv.org/abs/1711.00614",
            "description": "Variational autoencoder with a recurrent encoder-decoder, used to jointly model temporal dependencies and latent uncertainty in multivariate flight sensor sequences.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1-score for anomaly detection",
        "Area under the precision-recall curve (AUPRC)",
        "Reconstruction probability distribution separation between nominal and anomalous windows",
        "False positive rate stratified by flight phase",
    ],
    "implementation_steps": [
        "Curate multivariate flight sensor sequences (airspeed, altitude, engine parameters, control surface positions) from flight data recorder or quick access recorder archives",
        "Segment sequences by flight phase and normalize each sensor channel, handling missing or irregularly sampled readings",
        "Train a feedforward VAE and PCA/deterministic-autoencoder baselines on nominal sensor snapshots to establish reconstruction-based anomaly scores",
        "Train an LSTM-VAE on windowed sensor sequences to incorporate temporal structure, using reconstruction probability as the anomaly score",
        "Evaluate all models on held-out sequences containing injected or documented anomalies, comparing precision, recall, and AUPRC",
        "Analyze false positive rates separately by flight phase to assess whether normal operational variability is being mistaken for anomalies",
    ],
    "unverified_suggestions": [
        "NASA DASHlink flight data repository",
        "NASA C-MAPSS Turbofan Engine Degradation dataset",
        "NASA Advanced Diagnostics and Prognostics Testbed (ADAPT) dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal correctly distinguishes reconstruction-probability scoring from simple reconstruction error, grounding the anomaly scoring choice in the established VAE anomaly detection literature",
        "Including flight-phase-stratified false positive rate analysis directly addresses a realistic source of spurious alarms rather than assuming stationary sensor behavior throughout a flight",
        "Comparing against both PCA and deterministic autoencoder baselines gives a reasonably comprehensive spread of unsupervised comparison points",
        "The LSTM-VAE extension is well-motivated by the temporal nature of flight sensor data rather than applied as an unmotivated architectural upgrade",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how genuine anomalies will be labeled or injected for evaluation, given the acknowledged scarcity of real confirmed-anomaly flight data recorder events",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host flight data recorder sequences, so the project depends on the unverified NASA DASHlink and C-MAPSS suggestions, which vary in how closely they represent full-flight multivariate sensor behavior",
        "The plan does not address a genuinely important methodological consideration for this domain: latent space dimensionality selection directly trades off reconstruction fidelity against anomaly sensitivity, and no systematic sweep or selection criterion is specified",
    ],
    "missing_components": [
        "A systematic latent-dimensionality sensitivity analysis with a defined selection criterion rather than a single fixed latent size",
        "A discussion of how the safety-critical asymmetry between missed anomalies and false alarms will be reflected in the chosen detection threshold",
    ],
    "novelty_score": 0.63,
    "quality_score": 0.74,
}

CONFIDENCE_JSON = {"internal_consistency": 0.81, "specificity": 0.76, "completeness": 0.84}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to variational autoencoder anomaly detection in aircraft sensor data, based on the retrieved abstract." for t in titles]
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
