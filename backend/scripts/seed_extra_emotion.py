import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Emotion Recognition from Multimodal Physiological Signals Using Deep Learning"

PLANNER_JSON = {
    "research_objectives": [
        "Design a deep learning architecture that fuses EEG, ECG/PPG, EDA, and respiration signals for continuous emotion recognition",
        "Evaluate temporal and cross-modal fusion strategies for capturing complementary affective information across physiological channels",
        "Assess the generalization of multimodal emotion recognition models across subjects and recording sessions",
        "Investigate the trade-off between recognition accuracy and the number/type of physiological modalities used",
    ],
    "research_questions": [
        "How does fusing central (EEG) and peripheral (ECG, EDA, respiration) physiological signals improve emotion classification accuracy over single-modality models?",
        "Which fusion strategy (feature-level, decision-level, or attention-based) best captures complementary affective cues across physiological modalities?",
        "How well do multimodal physiological emotion recognition models generalize to unseen subjects in a leave-one-subject-out evaluation?",
        "What is the minimal combination of physiological modalities that preserves recognition accuracy while reducing sensor and computational overhead?",
    ],
    "keywords": [
        "multimodal emotion recognition",
        "physiological signal fusion",
        "EEG affective computing",
        "deep learning for biosignals",
        "cross-subject generalization",
    ],
    "research_scope": "This research focuses on subject-dependent and subject-independent emotion recognition from multimodal physiological signals, primarily EEG combined with peripheral signals such as ECG/PPG, electrodermal activity (EDA), and respiration, using publicly available affective computing datasets recorded under controlled stimulus-presentation protocols (e.g., video or music elicitation). The scope excludes facial expression, speech, and text-based emotion recognition modalities, real-time wearable deployment constraints, and clinical or diagnostic applications, concentrating instead on offline deep learning architectures for discrete or dimensional (valence-arousal) emotion classification from physiological time-series data.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of deep architectures (CNNs, LSTMs, and graph neural networks) to model spatial-temporal dependencies in EEG channels for emotion recognition",
        "Growing adoption of attention-based and transformer fusion mechanisms to dynamically weight the contribution of each physiological modality",
        "Rising interest in domain adaptation and contrastive learning techniques to address inter-subject variability in physiological signals",
        "Emerging focus on lightweight and interpretable models suitable for wearable-based affective computing applications",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of multimodal fusion robustness when one or more physiological channels are missing or degraded due to sensor artifacts, which commonly occurs in real-world wearable recordings.",
            "based_on_papers": ["A Survey on Emotion Recognition Using Physiological Signals"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-dataset generalization studies that train on one physiological emotion dataset (e.g., DEAP) and evaluate on another (e.g., WESAD or AMIGOS) to test true domain transfer rather than within-dataset splits.",
            "based_on_papers": ["Multimodal Emotion Recognition Using Deep Learning Architectures", "EEG-Based Emotion Recognition: A Review of Deep Learning Approaches"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative comparison of fusion architectures under varying stimulus modalities (video versus music elicitation) to determine whether fusion strategy effectiveness depends on how emotions were induced.",
            "based_on_papers": ["Attention-Based Multimodal Fusion for Affective Computing"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop cross-dataset and cross-device benchmarking protocols to rigorously test the generalization of multimodal physiological emotion recognition models",
        "Explore modality-dropout-robust and missing-data-tolerant fusion architectures suitable for real-world wearable sensor conditions",
        "Investigate self-supervised and contrastive pretraining on unlabeled physiological data to reduce reliance on scarce labeled affective datasets",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "PhysioNet", "url": "https://physionet.org/", "description": "Curated clinical and physiological signal datasets for healthcare research.", "verified": True}],
    "algorithms": [
        {
            "name": "Long Short-Term Memory (LSTM)",
            "url": "https://www.bioinf.jku.at/publications/older/2604.pdf",
            "description": "Recurrent neural network architecture with gated memory cells, used here to model temporal dynamics within each physiological signal stream (e.g., EDA and respiration sequences) before fusion.",
            "verified": True,
        },
        {
            "name": "Graph Convolutional Network (GCN) for EEG",
            "url": "https://arxiv.org/abs/1609.02907",
            "description": "Graph-based neural network that models EEG electrodes as nodes with functional connectivity as edges, capturing spatial dependencies across brain regions for emotion-relevant feature extraction.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Accuracy",
        "Macro F1-score",
        "Concordance Correlation Coefficient (for valence-arousal regression)",
        "Leave-One-Subject-Out cross-validation accuracy",
    ],
    "implementation_steps": [
        "Preprocess raw EEG, ECG/PPG, EDA, and respiration signals using bandpass filtering, artifact removal, and per-subject normalization",
        "Segment continuous recordings into fixed-length windows aligned with emotion labels (discrete categories or continuous valence-arousal ratings)",
        "Extract modality-specific features using CNN or LSTM encoders, and construct EEG channel graphs for GCN-based spatial feature extraction",
        "Implement and compare feature-level concatenation fusion, decision-level (late) fusion, and cross-modal attention fusion architectures",
        "Train models under both subject-dependent (stratified k-fold) and subject-independent (leave-one-subject-out) evaluation protocols",
        "Evaluate models using accuracy, macro F1, and concordance correlation coefficient, and conduct ablations removing individual modalities to assess their contribution",
    ],
    "unverified_suggestions": ["DEAP (Database for Emotion Analysis using Physiological signals)", "WESAD (Wearable Stress and Affect Detection) Dataset", "AMIGOS Multimodal Emotion Dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "Research objectives directly connect fusion architecture choices to measurable emotion recognition outcomes across multiple physiological modalities",
        "Methodology proposes a principled comparison of feature-level, decision-level, and attention-based fusion rather than committing to a single fusion strategy upfront",
        "Includes both subject-dependent and subject-independent (leave-one-subject-out) evaluation, addressing a well-known generalization challenge in physiological emotion recognition",
        "Incorporates modality ablation studies, providing insight into the marginal contribution of each physiological signal to overall performance",
    ],
    "weaknesses": [
        "The proposal does not specify how class imbalance across emotion categories (e.g., fewer high-arousal negative samples) will be handled during training",
        "Evidence sufficiency is limited: the identified gaps rely on a small set of 8 retrieved papers, so claims about underexplored areas like cross-dataset generalization should be treated as preliminary rather than exhaustive",
        "Dataset fit caveats: the suggested unverified datasets (e.g., DEAP, AMIGOS) require signed data-use agreements and have differing sensor configurations, which could complicate direct cross-dataset comparison and were not validated against the whitelist",
        "No explicit description of how synchronization across physiological modalities with differing sampling rates (e.g., EEG at 256 Hz versus respiration at 32 Hz) will be handled before fusion",
    ],
    "missing_components": [
        "A concrete plan for hyperparameter tuning and computational budget across the three fusion architectures and both evaluation protocols",
        "An error analysis section outlining how misclassified emotion states (e.g., confusion between high-arousal fear and excitement) will be examined",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to multimodal physiological emotion recognition using deep learning, based on the retrieved abstract." for t in titles]
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
