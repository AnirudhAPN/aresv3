import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Utility Meter Reading from Images"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning pipelines that automatically localize and read numeric digit sequences from photographs of electricity, water, and gas utility meters",
        "Evaluate convolutional recurrent architectures against segmentation-then-classification pipelines for end-to-end digit sequence recognition accuracy on meter images",
        "Investigate robustness to real-world capture conditions such as glare, dirt, partial occlusion of the digit display, and varying meter dial types (digital versus analog dial)",
        "Assess the practical deployment accuracy needed to replace manual meter reading, including error tolerance for billing-relevant digit misreads",
    ],
    "research_questions": [
        "Do convolutional recurrent architectures that read the full digit sequence end-to-end outperform pipelines that first segment individual digits and then classify each separately?",
        "How does meter-reading accuracy degrade under common real-world capture conditions such as glare, dirt accumulation, and partial occlusion of the display?",
        "Can a single model generalize across different meter types (digital counter, analog dial, smart meter LCD) without separate specialized models for each?",
        "What digit-level error rate is acceptable for automated meter reading to be considered reliable enough to replace manual billing-relevant readings?",
    ],
    "keywords": [
        "automatic meter reading",
        "optical character recognition digit sequence",
        "utility meter image recognition",
        "convolutional recurrent neural network",
        "smart metering computer vision",
    ],
    "research_scope": (
        "This research covers deep learning methods for automatically reading numeric digit sequences from "
        "photographs of electricity, water, and gas utility meters, including digit localization, segmentation, and "
        "sequence recognition, along with robustness to real-world capture artifacts and generalization across meter "
        "dial types. It excludes smart-meter wireless telemetry data collection that does not involve image capture, "
        "and it excludes non-numeric meter status indicators unrelated to the digit-reading task."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of convolutional recurrent neural network architectures that read full digit sequences end-to-end rather than segmenting and classifying digits independently",
        "Increasing use of synthetic and augmented training data (simulated glare, dirt, occlusion) to improve robustness without requiring large real-world annotated capture datasets",
        "Rising interest in multi-meter-type models capable of handling digital counter, analog dial, and smart-meter LCD displays within a single pipeline",
        "Emerging focus on deployment-oriented evaluation metrics that weigh billing-relevant digit errors more heavily than cosmetic misreads",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of end-to-end sequence recognition versus segment-then-classify pipelines under matched real-world capture conditions.",
            "based_on_papers": ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic robustness benchmarking across combined artifact types (glare plus dirt plus partial occlusion simultaneously), since most studies test artifacts independently.",
            "based_on_papers": ["A Public Dataset for Automatic Meter Reading"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: a single unified model evaluated across digital, analog dial, and smart-meter LCD display types, as most existing work targets one meter type at a time.",
            "based_on_papers": ["A Public Dataset for Automatic Meter Reading"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build combined-artifact robustness benchmarks that jointly test glare, dirt, and occlusion conditions rather than in isolation",
        "Develop and evaluate unified multi-meter-type models spanning digital, analog dial, and smart-meter displays",
        "Establish deployment-oriented evaluation protocols that weight billing-relevant digit errors according to their financial impact",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "ResNet",
            "url": "https://arxiv.org/abs/1512.03385",
            "description": "Deep residual convolutional network used to localize the meter display region and extract visual features from meter images.",
            "verified": True,
        },
        {
            "name": "CRNN (Convolutional Recurrent Neural Network)",
            "url": "https://arxiv.org/abs/1507.05717",
            "description": "End-to-end trainable architecture combining convolutional feature extraction with recurrent sequence modeling, used to read the full digit sequence from the localized meter display without explicit per-digit segmentation.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Digit sequence exact-match accuracy",
        "Per-digit character error rate",
        "Accuracy stratified by capture artifact (glare, dirt, occlusion)",
        "Accuracy stratified by meter dial type (digital, analog, smart-meter LCD)",
    ],
    "implementation_steps": [
        "Assemble and annotate a dataset of utility meter photographs with ground-truth digit sequences, labeling meter type and visible capture artifacts where present",
        "Implement a ResNet-based detector to localize the digit display region within each meter photograph",
        "Implement a CRNN model to read the full digit sequence from the localized display region, alongside a segment-then-classify baseline pipeline",
        "Apply data augmentation simulating glare, dirt, and partial occlusion to improve robustness, and evaluate on held-out real images exhibiting these artifacts",
        "Train and evaluate a unified model across all available meter dial types, comparing against dial-type-specialized models",
        "Analyze billing-relevant error cases (misread digits that would change a customer's bill) separately from cosmetically incorrect but billing-neutral errors",
    ],
    "unverified_suggestions": ["UFPR-AMR (UFPR Automatic Meter Reading) dataset", "Copel-AMR dataset", "Watermeter/electric meter image dataset (Kaggle)"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds its architecture comparison in a deployment-relevant question (end-to-end versus segment-then-classify) rather than benchmarking novelty for its own sake",
        "Explicitly stratifying evaluation by capture artifact and meter dial type gives a much more actionable diagnostic than a single aggregate accuracy figure",
        "Distinguishing billing-relevant digit errors from cosmetically incorrect but financially neutral errors reflects real utility-industry priorities",
        "The unified multi-meter-type modeling objective addresses a genuine practical need for utilities managing mixed meter fleets",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) does not typically host utility meter image datasets, so UFPR-AMR, Copel-AMR, and similar sources remain unverified suggestions requiring separate access.",
        "The proposal does not address privacy or data-handling considerations for meter photographs, which can incidentally capture other property details or personal information visible in the frame.",
        "There is limited discussion of how the system would flag low-confidence readings for human review rather than silently propagating an incorrect automated reading into a customer's bill.",
    ],
    "missing_components": [
        "A privacy and data-handling protocol for meter photographs that may incidentally capture extraneous visual information.",
        "A confidence-based human-in-the-loop review mechanism for low-certainty automated readings before they affect billing.",
    ],
    "novelty_score": 0.61,
    "quality_score": 0.70,
}

CONFIDENCE_JSON = {"internal_consistency": 0.75, "specificity": 0.77, "completeness": 0.78}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to automated utility meter reading from images, based on the retrieved abstract." for t in titles]
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
