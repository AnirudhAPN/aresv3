import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Bioacoustic Deep Learning for Automated Wildlife Species Identification"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning pipeline that automatically detects and classifies wildlife species from passive acoustic monitoring recordings",
        "Evaluate spectrogram-based convolutional and recurrent architectures for robustness to background noise, overlapping vocalizations, and variable recording conditions",
        "Assess the extent to which transfer learning from large-scale bioacoustic corpora improves species identification accuracy for data-scarce or endangered species",
        "Investigate methods for reducing false positive detections caused by anthropogenic noise, wind, and rain in field-deployed acoustic sensors",
    ],
    "research_questions": [
        "Can convolutional neural networks trained on mel-spectrogram representations reliably distinguish between acoustically similar wildlife species in noisy field recordings?",
        "How does model performance degrade when moving from curated benchmark datasets to raw passive acoustic monitoring data collected in the field?",
        "What is the minimum amount of labeled species-specific audio required to fine-tune a pretrained bioacoustic model to acceptable accuracy for a new or rare species?",
        "Which architectural choices (CNN, CRNN, or transformer-based audio encoders) best balance classification accuracy against inference latency for edge deployment on autonomous recording units?",
    ],
    "keywords": [
        "bioacoustic monitoring",
        "wildlife species identification",
        "convolutional neural networks",
        "passive acoustic monitoring",
        "audio spectrogram classification",
    ],
    "research_scope": "This research focuses on designing and evaluating deep learning models for automated species identification from passive acoustic monitoring (PAM) recordings, spanning birds, anurans, and mammals with distinctive vocalizations. The scope covers preprocessing of raw audio into spectrogram or embedding representations, architecture selection (CNNs, convolutional recurrent networks, and pretrained audio transformers), transfer learning strategies for low-resource species, and evaluation under realistic field noise conditions such as wind, rain, insect chorus, and human activity. It also considers deployment constraints for battery-powered autonomous recording units at the edge. The scope excludes real-time individual animal re-identification and non-acoustic sensing modalities such as camera traps, though multimodal fusion with such sensors is noted as a future direction.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of CNNs applied to mel-spectrogram or log-mel representations for bird and anuran call classification, exemplified by tools such as BirdNET and Merlin Sound ID",
        "Growing use of large pretrained audio embedding models (e.g., VGGish, PANNs, and animal-specific foundation models) as fixed feature extractors for downstream species classifiers",
        "Increasing deployment of low-cost autonomous recording units (ARUs) generating continuous multi-month acoustic datasets that outstrip manual annotation capacity",
        "Emerging interest in self-supervised and few-shot learning approaches to handle the long-tailed distribution of species with very few labeled vocalization examples",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of how model accuracy degrades from clean, curated training recordings to raw multi-day ARU deployments containing overlapping calls, wind noise, and rain, with quantified performance gaps rather than anecdotal field validation.",
            "based_on_papers": ["BirdNET: A deep learning solution for avian diversity monitoring"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: limited exploration of few-shot or metric-learning approaches specifically tailored to endangered species for which only a handful of labeled vocalizations exist, as most benchmarks assume hundreds to thousands of examples per class.",
            "based_on_papers": ["Deep learning for bioacoustic monitoring"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: insufficient attention to on-device inference constraints (memory, power, latency) for deploying trained species classifiers directly on autonomous recording units rather than relying on post-hoc cloud processing.",
            "based_on_papers": ["Acoustic monitoring of biodiversity using machine learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop domain adaptation techniques that explicitly bridge the gap between curated training corpora and heterogeneous field-recorded acoustic conditions",
        "Design few-shot and metric-learning classifiers that can be rapidly adapted to newly monitored or endangered species with minimal labeled data",
        "Explore lightweight, quantized model architectures suitable for real-time on-device species identification on battery-constrained autonomous recording units",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "Kaggle Datasets",
            "url": "https://www.kaggle.com/datasets",
            "description": "Community-contributed datasets spanning nearly every domain, with notebooks.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
            "description": "Backbone architecture applied to mel-spectrogram representations of audio clips to learn discriminative time-frequency patterns for species classification.",
            "verified": True,
        },
        {
            "name": "Convolutional Recurrent Neural Network (CRNN)",
            "url": "https://arxiv.org/abs/1609.04243",
            "description": "Hybrid architecture combining convolutional layers for local spectrogram feature extraction with recurrent layers to model the temporal structure of extended or sequential animal vocalizations.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision",
        "Recall",
        "F1-score",
        "Mean Average Precision (mAP)",
    ],
    "implementation_steps": [
        "Assemble and clean a labeled corpus of species vocalizations from public repositories such as Xeno-canto and Macaulay Library, supplemented with field ARU recordings",
        "Segment continuous audio into fixed-length clips and convert each clip into log-mel spectrogram representations with consistent time-frequency resolution",
        "Apply data augmentation (time-shifting, pitch shifting, background noise mixing, and SpecAugment masking) to improve robustness to field recording conditions",
        "Train and compare CNN and CRNN classifiers, optionally initialized from pretrained bioacoustic embeddings, using stratified splits to handle class imbalance across species",
        "Evaluate models on held-out field-recorded ARU data separate from the curated training corpus to measure real-world generalization",
        "Benchmark inference latency and memory footprint of the best-performing model for feasibility of on-device deployment on autonomous recording units",
    ],
    "unverified_suggestions": [
        "Consider incorporating self-supervised pretraining on large volumes of unlabeled ARU audio to improve representations for rare species with few labels",
        "Explore transformer-based audio encoders (e.g., Audio Spectrogram Transformer) as an alternative to CNN/CRNN backbones for capturing longer-range temporal dependencies in call sequences",
        "Investigate active learning strategies to prioritize which unlabeled field recordings are most valuable for expert annotation given limited labeling budgets",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates automated species identification as a scalable alternative to manual review of continuous multi-month acoustic monitoring data",
        "The methodology grounds preprocessing and augmentation choices in field-realistic noise conditions rather than assuming clean benchmark audio",
        "Including on-device latency and memory benchmarking demonstrates awareness of practical deployment constraints for edge-based conservation monitoring",
        "Evaluation on held-out field ARU data, separate from the curated training corpus, directly tests real-world generalization rather than only benchmark accuracy",
    ],
    "weaknesses": [
        "The evidence base is limited to 8 retrieved papers, which may not sufficiently cover recent few-shot or self-supervised bioacoustic methods published in the past two years",
        "The proposal does not specify how it will handle overlapping vocalizations from multiple species calling simultaneously, a common and difficult scenario in dense habitats",
        "No baseline comparison against existing widely used tools such as BirdNET is planned, making it difficult to contextualize claimed improvements",
        "The dataset composition across species is not addressed, and severe class imbalance between common and rare species could bias reported aggregate metrics",
    ],
    "missing_components": [
        "An explicit plan for handling overlapping or simultaneous multi-species vocalizations during both training and evaluation",
        "A discussion of how annotation quality and inter-annotator agreement on species labels will be verified for the training corpus",
    ],
    "novelty_score": 0.66,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to bioacoustic deep learning for automated wildlife species identification, based on the retrieved abstract." for t in titles]
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
