import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Self-Supervised Video Representation Learning for Surgical Skill Assessment"

PLANNER_JSON = {
    "research_objectives": [
        "Develop self-supervised video representation learning methods that learn surgical action- and skill-relevant features from unlabeled surgical video",
        "Evaluate contrastive and masked-video-modeling pretraining objectives for downstream surgical skill classification and scoring tasks",
        "Investigate how well self-supervised pretrained representations transfer across different surgical procedures and skill assessment rubrics such as OSATS and GEARS",
        "Assess the temporal modeling requirements, comparing short gesture-level clips against long full-procedure video, for accurately capturing skill-relevant motion patterns",
    ],
    "research_questions": [
        "Can self-supervised pretraining on unlabeled surgical video improve downstream skill assessment accuracy compared to training from scratch or using ImageNet-pretrained features?",
        "How do contrastive pretraining objectives compare to masked video modeling objectives for capturing fine-grained motion cues relevant to surgical skill?",
        "How well do representations pretrained on one surgical procedure type transfer to skill assessment on a different procedure?",
        "What temporal context length, short gesture-level clips versus full-procedure video, is necessary for self-supervised representations to capture clinically meaningful skill differences?",
    ],
    "keywords": [
        "self-supervised video representation learning surgery",
        "surgical skill assessment deep learning",
        "video contrastive learning surgical video",
        "masked video modeling VideoMAE",
        "surgical gesture recognition JIGSAWS",
    ],
    "research_scope": (
        "This research focuses on self-supervised pretraining of video representations from surgical procedure "
        "recordings, and their transfer to downstream automated surgical skill assessment tasks scored against "
        "established rubrics such as OSATS and GEARS. It covers contrastive and masked-video-modeling pretraining "
        "objectives, cross-procedure transfer evaluation, and analysis of the temporal context window needed for "
        "skill-relevant representation learning. The scope excludes real-time intraoperative decision support, "
        "surgical instrument tracking as a standalone task, and robotic surgical control, which are treated as "
        "related but separate research problems outside this work."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing application of self-supervised video pretraining objectives, including contrastive learning and masked video modeling, originally developed for general action recognition, to surgical video analysis",
        "Use of the JIGSAWS dataset and OSATS/GEARS rating scales as standard benchmarks for automated surgical skill assessment",
        "Adoption of temporal segment or gesture-level modeling to align learned representations with clinically meaningful surgical sub-tasks",
        "Interest in reducing reliance on scarce expert-annotated skill labels by leveraging large volumes of unlabeled surgical video from operating room recordings",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of contrastive versus masked video modeling self-supervised objectives specifically for downstream surgical skill assessment performance.",
            "based_on_papers": ["A Simple Framework for Contrastive Learning of Visual Representations"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-procedure transfer evaluation of self-supervised surgical video representations, as most existing work benchmarks within a single procedure type such as JIGSAWS suturing tasks.",
            "based_on_papers": ["JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS)"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of the temporal context window needed for self-supervised representations to capture clinically meaningful skill differences, as opposed to short-clip action-recognition-style pretraining windows.",
            "based_on_papers": ["VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish controlled ablation studies directly comparing contrastive and masked-video-modeling objectives on identical surgical skill assessment downstream tasks",
        "Develop multi-procedure surgical video benchmarks explicitly designed to evaluate cross-procedure representation transfer",
        "Investigate adaptive or hierarchical temporal windowing that captures both short gesture-level and long procedure-level skill cues",
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
            "name": "VideoMAE",
            "url": "https://arxiv.org/abs/2203.12602",
            "description": "Masked video autoencoder that reconstructs heavily masked spatiotemporal patches, used as a self-supervised pretraining objective for learning motion-sensitive representations from unlabeled surgical video.",
            "verified": True,
        },
        {
            "name": "SimCLR",
            "url": "https://arxiv.org/abs/2002.05709",
            "description": "Contrastive learning framework that maximizes agreement between augmented views of the same clip, adapted to video and used as an alternative self-supervised pretraining objective for comparison against masked video modeling.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "OSATS/GEARS score prediction correlation (Pearson/Spearman)",
        "Skill classification accuracy (novice/intermediate/expert)",
        "Mean absolute error (MAE) on predicted skill score",
        "Downstream linear-probe accuracy",
    ],
    "implementation_steps": [
        "Collect unlabeled surgical procedure video, such as from JIGSAWS or institutional operating room recordings, for self-supervised pretraining",
        "Apply a contrastive or masked-video-modeling pretraining objective on unlabeled surgical video clips",
        "Fine-tune or linear-probe the pretrained encoder on labeled skill assessment data annotated with OSATS/GEARS scores",
        "Benchmark against ImageNet-pretrained and from-scratch trained baselines on the same downstream skill assessment task",
        "Evaluate cross-procedure transfer by pretraining on one procedure type and fine-tuning/testing on a different procedure",
        "Analyze the effect of temporal clip length, from short gesture-level segments to full-procedure video, on downstream skill prediction accuracy",
    ],
    "unverified_suggestions": ["JIGSAWS (JHU-ISI Gesture and Skill Assessment Working Set)", "Cholec80 laparoscopic video dataset", "SurgicalActions160"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly addresses the practical constraint of scarce expert-annotated skill labels by centering the design around self-supervised pretraining on unlabeled video",
        "Comparing contrastive and masked-video-modeling objectives head-to-head on the same downstream task provides a clear, well-motivated ablation missing from much prior work",
        "The cross-procedure transfer evaluation directly tests generalization beyond the single-procedure JIGSAWS benchmark that dominates the field",
        "Including both correlation-based and classification-based skill metrics (OSATS/GEARS correlation and novice/intermediate/expert accuracy) gives a more complete picture than either alone",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how institutional OR recordings, if used, will satisfy patient privacy, de-identification, and institutional review board requirements before being used for model training",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host surgical video data, so feasibility depends entirely on the unverified, access-restricted dataset suggestions such as JIGSAWS and Cholec80",
        "The plan does not address inter-rater variability in OSATS/GEARS expert scoring itself, which is a known source of label noise that could confound comparisons between pretraining objectives if not explicitly modeled or reported",
    ],
    "missing_components": [
        "A data governance and IRB/privacy compliance plan for any institutional operating room video used in pretraining",
        "An analysis of inter-rater agreement among expert OSATS/GEARS scorers and its impact on the reliability of downstream evaluation labels",
    ],
    "novelty_score": 0.73,
    "quality_score": 0.7,
}

CONFIDENCE_JSON = {"internal_consistency": 0.77, "specificity": 0.75, "completeness": 0.8}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to self-supervised video representation learning for surgical skill assessment, based on the retrieved abstract." for t in titles]
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
