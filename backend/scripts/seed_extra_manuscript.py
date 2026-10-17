import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Handwriting Recognition in Historical Manuscripts"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning pipelines for offline handwriting recognition of historical manuscript images, covering line/word segmentation and text transcription",
        "Evaluate convolutional recurrent neural network (CRNN) and transformer-based OCR architectures on degraded, faded, and archaic-script manuscript images",
        "Investigate transfer learning strategies that adapt models pretrained on modern handwriting corpora to historical scripts with limited labeled manuscript data",
        "Assess the impact of manuscript-specific degradations (ink bleed-through, faded text, non-standard orthography) on recognition accuracy compared to modern handwriting benchmarks",
    ],
    "research_questions": [
        "How do CRNN-based and transformer-based OCR architectures compare on transcription accuracy for historical manuscript images relative to modern handwriting benchmarks?",
        "What transfer learning strategies most effectively adapt models pretrained on modern handwriting datasets to historical scripts with limited labeled data?",
        "How much does manuscript-specific image degradation (bleed-through, fading, staining) degrade recognition accuracy, and which preprocessing steps mitigate this most effectively?",
        "How well do models trained on one historical script or scribal hand generalize to a different scribal hand or manuscript collection?",
    ],
    "keywords": [
        "historical manuscript handwriting recognition",
        "offline handwritten text recognition",
        "CRNN optical character recognition",
        "transformer OCR historical documents",
        "transfer learning handwriting transcription",
    ],
    "research_scope": (
        "This research focuses on deep learning methods for offline handwriting recognition applied to images of "
        "historical manuscripts, covering text line and word segmentation, CRNN and transformer-based transcription "
        "architectures, and transfer learning from modern handwriting corpora to historical scripts affected by "
        "degradation and archaic orthography. It excludes printed text OCR, online handwriting recognition captured "
        "via stylus or digital pen input, and manuscript dating or authorship attribution, which are treated as out "
        "of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of CRNN architectures combining convolutional feature extraction with recurrent sequence modeling for offline handwritten text recognition",
        "Increasing use of transformer-based OCR models pretrained on large synthetic and modern handwriting corpora, then fine-tuned for specialized domains",
        "Continued reliance on a small number of curated historical manuscript benchmarks (e.g., READ and Bentham-style collections) due to the cost of expert transcription",
        "Rising interest in transfer learning and domain adaptation to reduce the labeled-data requirements for each new historical script or scribal hand",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic study of cross-scribe generalization, i.e., how much recognition accuracy degrades when a model trained on one scribal hand or manuscript collection is applied to a distinct, unseen scribal hand.",
            "based_on_papers": ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of CRNN and transformer-based OCR architectures under matched training data size specifically on historical manuscript images, rather than modern handwriting or printed text benchmarks.",
            "based_on_papers": ["TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative ablation isolating the effect of specific manuscript degradations (ink bleed-through, fading, staining) on recognition accuracy, as opposed to reporting only aggregate transcription error rates on already-curated benchmark images.",
            "based_on_papers": ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop few-shot or meta-learning approaches that adapt handwriting recognition models to a new scribal hand from only a handful of labeled manuscript lines",
        "Build degradation-specific augmentation pipelines and paired evaluation subsets to isolate the impact of bleed-through, fading, and staining on transcription accuracy",
        "Create standardized cross-collection benchmarks spanning multiple historical scripts and scribal hands to enable fairer comparison of CRNN and transformer-based approaches",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "CRNN (Convolutional Recurrent Neural Network)",
            "url": "https://arxiv.org/abs/1507.05717",
            "description": "End-to-end trainable architecture combining convolutional feature extraction with recurrent sequence modeling and CTC decoding, used as the baseline transcription model for manuscript line images.",
            "verified": True,
        },
        {
            "name": "TrOCR",
            "url": "https://arxiv.org/abs/2109.10282",
            "description": "Transformer-based OCR model using a pretrained image transformer encoder and text transformer decoder, used as the primary architecture to evaluate transfer learning from modern to historical handwriting.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Character Error Rate (CER)",
        "Word Error Rate (WER)",
        "Line-level transcription accuracy",
        "Accuracy stratified by degradation type (bleed-through, fading, staining)",
    ],
    "implementation_steps": [
        "Collect and digitize historical manuscript page images, performing text line and word segmentation to produce aligned image-transcription pairs",
        "Annotate or curate a labeled subset with expert transcriptions, stratifying examples by degradation type and scribal hand where metadata is available",
        "Pretrain or initialize CRNN and TrOCR models on modern handwriting corpora, then fine-tune on the labeled historical manuscript subset",
        "Apply degradation-aware data augmentation (simulated fading, bleed-through, staining) to the fine-tuning set to improve robustness",
        "Evaluate both architectures on held-out manuscript pages using character error rate, word error rate, and degradation-stratified accuracy",
        "Test cross-scribe generalization by evaluating on a scribal hand or manuscript collection not represented in the fine-tuning data",
    ],
    "unverified_suggestions": [
        "IAM Handwriting Database",
        "READ dataset (Recognition and Enrichment of Archival Documents)",
        "Bentham Papers historical manuscript dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal explicitly targets cross-scribe generalization, a realistic and practically important failure mode rather than assuming a single scribal hand covers the deployment scenario",
        "Combining transfer learning from modern handwriting corpora with degradation-aware augmentation is a sound strategy for the well-known label scarcity problem in historical manuscript datasets",
        "Stratifying evaluation by degradation type (bleed-through, fading, staining) gives a genuinely diagnostic breakdown rather than a single aggregate error rate",
        "Comparing CRNN and transformer-based (TrOCR) architectures under matched fine-tuning data size directly targets a real, underexplored empirical question for this domain",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how expert transcription annotations will be quality-controlled, given that historical paleography interpretation can vary between experts",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host manuscript image-transcription pairs, so the project depends on the unverified IAM, READ, and Bentham dataset suggestions, whose language, script, and period coverage may not match the target manuscripts",
        "The plan does not address a genuinely important methodological consideration for this domain: archaic orthography and abbreviation conventions (e.g., historical ligatures and scribal abbreviations) require a different, non-standard character/token vocabulary than modern handwriting datasets provide, and this is not discussed",
    ],
    "missing_components": [
        "A plan for defining or adapting a transcription vocabulary that accounts for archaic ligatures, abbreviations, and orthographic conventions specific to the target manuscripts",
        "A discussion of inter-annotator agreement or quality control for the expert transcription ground truth used in fine-tuning and evaluation",
    ],
    "novelty_score": 0.61,
    "quality_score": 0.71,
}

CONFIDENCE_JSON = {"internal_consistency": 0.78, "specificity": 0.75, "completeness": 0.80}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning for handwriting recognition in historical manuscripts, based on the retrieved abstract." for t in titles]
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
