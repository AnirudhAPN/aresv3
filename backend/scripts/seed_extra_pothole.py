import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Pothole Detection Using Smartphone Accelerometer Data"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning classifiers that detect potholes and road surface anomalies from smartphone accelerometer and gyroscope time series collected during ordinary vehicle travel",
        "Evaluate LSTM and 1D convolutional neural network architectures against classical threshold-based and feature-engineered machine learning baselines for pothole event classification",
        "Investigate the effect of smartphone mounting position, vehicle type, and driving speed on accelerometer signal characteristics and downstream classification accuracy",
        "Assess the feasibility of crowd-sourced, GPS-tagged pothole detection for road-quality mapping using only commodity smartphone inertial sensors",
    ],
    "research_questions": [
        "Can deep learning models classify pothole events from smartphone accelerometer and gyroscope time series more accurately than threshold-based or hand-engineered feature baselines?",
        "How do LSTM-based and 1D-CNN architectures compare in accuracy and inference latency for real-time, on-device pothole event classification?",
        "How much does classification accuracy degrade when smartphone mounting position, vehicle type, or driving speed differs from the conditions used during training?",
        "What GPS-tagging and event-aggregation strategy best converts individual detected pothole events into a reliable, low-false-positive road-quality map from crowd-sourced data?",
    ],
    "keywords": [
        "pothole detection accelerometer",
        "smartphone sensor road anomaly detection",
        "deep learning vehicle vibration classification",
        "crowdsourced road quality mapping",
        "time series classification inertial sensors",
    ],
    "research_scope": (
        "This research focuses on deep learning classification of road potholes and surface anomalies from "
        "smartphone inertial sensor streams (accelerometer and gyroscope) collected during vehicle travel, covering "
        "time-series segmentation, LSTM and 1D-CNN classifier architectures, and GPS-tagged crowd-sourced "
        "aggregation for road-quality mapping. It includes evaluation of robustness to mounting position, vehicle "
        "type, and driving speed. The scope excludes vision-based pothole detection from dashboard or roadside "
        "cameras, LIDAR-based road surface scanning, and pavement material or structural engineering analysis, "
        "which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Continued reference to early smartphone-sensor road-anomaly systems such as Nericell and subsequent accelerometer-based pothole detection studies as foundational baselines",
        "Growing use of LSTM and 1D convolutional neural networks for accelerometer time-series classification, replacing earlier hand-engineered feature and threshold-based detectors",
        "Increasing interest in crowd-sourced, GPS-tagged sensing as a low-cost alternative to dedicated road-inspection vehicles for large-scale road-quality mapping",
        "Persistent variability in reported accuracy across studies due to differing smartphone mounting positions, vehicle suspension characteristics, and driving speeds",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of classification robustness when a model trained with one smartphone mounting position (e.g., dashboard-mounted) is tested with a different mounting position (e.g., in a pocket or cup holder).",
            "based_on_papers": ["Real Time Pothole Detection Using Android Smartphones with Accelerometers"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of LSTM-based and 1D-CNN architectures under matched training data and driving conditions for on-device, real-time pothole classification.",
            "based_on_papers": ["Nericell: Rich Monitoring of Road and Traffic Conditions using Mobile Smartphones"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of false-positive rates introduced by non-pothole road events (speed bumps, railway crossings, potholes versus manhole covers) in crowd-sourced aggregation pipelines, as opposed to controlled single-vehicle test-drive evaluations.",
            "based_on_papers": ["Real Time Pothole Detection Using Android Smartphones with Accelerometers"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop mounting-position-invariant feature representations or explicit calibration steps to improve robustness across differing smartphone placements",
        "Establish shared benchmarks directly comparing LSTM and 1D-CNN architectures for on-device inertial sensor classification under matched conditions",
        "Design confusion-aware aggregation pipelines that explicitly model and filter common false-positive road events (speed bumps, manhole covers, railway crossings) in crowd-sourced pothole maps",
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
            "description": "Recurrent architecture used to classify pothole and road-anomaly events from sequential accelerometer and gyroscope windows.",
            "verified": True,
        },
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "description": "1D convolutional architecture applied directly to raw or lightly preprocessed inertial sensor windows, used as a faster on-device alternative to the recurrent LSTM classifier.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1-score for pothole event classification",
        "False positive rate on non-pothole road events (speed bumps, manhole covers, railway crossings)",
        "On-device inference latency",
        "Localization accuracy of GPS-tagged detected events against ground-truth pothole locations",
    ],
    "implementation_steps": [
        "Collect accelerometer and gyroscope time series from smartphones mounted in multiple positions across different vehicles and driving speeds, with GPS-tagged ground-truth pothole locations",
        "Segment continuous sensor streams into fixed-length windows and label windows containing confirmed pothole events versus normal driving and other road features",
        "Train LSTM and 1D-CNN classifiers on the labeled windows, tuning for both classification accuracy and inference latency suitable for on-device deployment",
        "Benchmark against a threshold-based baseline using hand-engineered features (e.g., peak vertical acceleration, jerk) similar to early accelerometer-based systems",
        "Evaluate robustness by testing models on held-out mounting positions, vehicle types, and driving speeds not seen during training",
        "Aggregate GPS-tagged detected events across simulated crowd-sourced trips and evaluate resulting road-quality map accuracy and false-positive rate",
    ],
    "unverified_suggestions": [
        "UAH-DriveSet naturalistic driving dataset",
        "Nericell-style smartphone road-sensing dataset",
        "Kaggle road surface anomaly accelerometer dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly tests robustness to smartphone mounting position and vehicle type, addressing a well-known and practically critical limitation of prior accelerometer-based pothole detection systems",
        "Including on-device inference latency as an evaluation metric reflects a realistic understanding that this application must run in real time on commodity hardware, not just achieve high offline accuracy",
        "Benchmarking against a threshold-based feature-engineered baseline situates the deep learning contribution relative to established prior systems like Nericell rather than only comparing among neural architectures",
        "The false-positive analysis on confusable non-pothole events (speed bumps, manhole covers, railway crossings) targets a genuinely important practical failure mode for crowd-sourced deployment",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how ground-truth pothole labels and locations will be verified at scale, given that manual confirmation of each detected event is labor-intensive",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host smartphone inertial sensor road-anomaly recordings, so the project depends on the unverified UAH-DriveSet and Kaggle dataset suggestions, which vary in whether they include labeled pothole ground truth at all",
        "The plan does not address a genuinely important methodological consideration for this domain: differences in vehicle suspension stiffness across vehicle classes (e.g., sedans versus trucks) can systematically alter accelerometer signal amplitude for the same physical pothole, and this potential confound is not discussed",
    ],
    "missing_components": [
        "A plan for normalizing or calibrating accelerometer signal amplitude across differing vehicle suspension characteristics",
        "A discussion of privacy considerations for continuously collecting GPS-tagged smartphone sensor data from crowd-sourced participants",
    ],
    "novelty_score": 0.64,
    "quality_score": 0.73,
}

CONFIDENCE_JSON = {"internal_consistency": 0.79, "specificity": 0.76, "completeness": 0.82}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning for pothole detection using smartphone accelerometer data, based on the retrieved abstract." for t in titles]
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
