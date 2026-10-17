import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated ICD Code Assignment from Clinical Notes"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning pipeline for multi-label ICD code assignment directly from unstructured clinical discharge notes",
        "Evaluate transformer-based clinical language models against label-attention CNN/RNN baselines for extreme multi-label ICD coding",
        "Investigate label-attention and ICD hierarchy structure modeling (chapter/category relationships) to improve rare-code prediction performance",
        "Assess model calibration and error patterns to determine suitability for human-in-the-loop coding assistance versus fully automated coding",
    ],
    "research_questions": [
        "Can transformer-based clinical language models improve ICD code assignment accuracy over CNN/RNN label-attention baselines, particularly for rare/long-tail codes?",
        "How much does incorporating ICD hierarchy structure (chapter/category/subcategory relationships) improve prediction of infrequently occurring codes?",
        "What is the tradeoff between full-note-length transformer processing with long-context handling versus truncation/chunking strategies for very long discharge summaries?",
        "How well-calibrated are model confidence scores for supporting human coder review workflows rather than fully autonomous coding?",
    ],
    "keywords": [
        "ICD code assignment deep learning",
        "clinical text multi-label classification",
        "automated medical coding transformer",
        "ClinicalBERT MIMIC-III",
        "label attention clinical notes",
    ],
    "research_scope": (
        "This research focuses on automated multi-label ICD code assignment from unstructured clinical text such as "
        "discharge summaries, covering transformer-based clinical language models, label-attention CNN/RNN "
        "architectures, ICD hierarchy-aware modeling, and calibration analysis for human-in-the-loop review. It uses "
        "publicly available de-identified clinical text benchmarks. The scope excludes structured EHR data coding "
        "(e.g., from lab results or billing codes derived from structured fields), clinical note generation, and "
        "full revenue-cycle/medical-billing workflow integration, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread benchmarking on the MIMIC-III discharge summary dataset, with label-attention CNN/RNN models such as CAML serving as long-standing strong baselines",
        "Growing adoption of pretrained clinical transformer language models (ClinicalBERT, Bio_ClinicalBERT, Clinical Longformer) fine-tuned for ICD coding",
        "Increasing use of ICD hierarchy-aware architectures that exploit code co-occurrence and hierarchical relationships to improve long-tail code prediction",
        "Interest in handling very long clinical documents via hierarchical or sparse-attention transformers to overcome standard transformer context-length limits",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of long-context transformer architectures against chunk-and-aggregate strategies specifically for full discharge summary ICD coding.",
            "based_on_papers": ["ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative evaluation of prediction calibration and its practical adequacy for human-in-the-loop coding assistance workflows, rather than accuracy metrics alone.",
            "based_on_papers": ["Explainable Prediction of Medical Codes from Clinical Text"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: rare/long-tail ICD code performance improvement from hierarchy-aware modeling evaluated specifically on codes with very few training examples, rather than aggregate micro-averaged metrics that are dominated by common codes.",
            "based_on_papers": ["Explainable Prediction of Medical Codes from Clinical Text"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized benchmarks directly comparing long-context transformers against chunk-and-aggregate baselines on identical discharge summary splits",
        "Establish calibration evaluation protocols tailored to human-in-the-loop coding assistance rather than purely automated accuracy reporting",
        "Design evaluation reporting standards that break out rare/long-tail code performance separately from aggregate micro-averaged metrics",
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
            "name": "ClinicalBERT",
            "url": "https://arxiv.org/abs/1904.05342",
            "description": "BERT-based language model pretrained on clinical text, fine-tuned here for multi-label ICD code classification from discharge summaries.",
            "verified": True,
        },
        {
            "name": "CAML (Convolutional Attention for Multi-Label classification)",
            "url": "https://arxiv.org/abs/1802.05695",
            "description": "Label-attention convolutional architecture that learns per-code attention over the input text, used as a strong non-transformer baseline for ICD code assignment.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Micro-averaged F1",
        "Macro-averaged F1",
        "Precision@k (e.g., P@8)",
        "Area under the ROC curve (AUC) for top-k codes",
    ],
    "implementation_steps": [
        "Preprocess MIMIC-III discharge summaries and associated ICD-9/ICD-10 code labels, including de-identification verification",
        "Tokenize and chunk long documents to fit within transformer context windows, or apply a long-context architecture",
        "Fine-tune ClinicalBERT/Bio_ClinicalBERT for multi-label classification with a sigmoid output layer over the code space",
        "Implement the CAML label-attention baseline for direct comparison",
        "Incorporate ICD hierarchy information via graph-based label embeddings or a hierarchical loss term to improve rare-code prediction",
        "Evaluate micro/macro F1, precision@k, and calibration on a held-out test split, with separate reporting for the rare-code subset",
    ],
    "unverified_suggestions": ["MIMIC-III Clinical Database", "MIMIC-IV", "eICU Collaborative Research Database"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal explicitly separates aggregate performance from rare/long-tail code performance, addressing a well-known blind spot in prior ICD coding evaluations",
        "Including calibration analysis directly targets the practical human-in-the-loop deployment scenario rather than assuming full automation",
        "Comparing a transformer-based approach against the established CAML baseline provides a meaningful, literature-grounded point of reference",
        "The long-context versus chunk-and-aggregate comparison addresses a concrete architectural tradeoff specific to the long-document nature of discharge summaries",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify the data use agreement and credentialing process required to access MIMIC-III/MIMIC-IV, which is a nontrivial prerequisite for reproducibility by other researchers",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host clinical text or ICD-labeled datasets, so the project depends entirely on the unverified, credential-gated PhysioNet dataset suggestions",
        "The plan does not address potential label noise in the ICD codes themselves, which are billing-derived and known to contain systematic coding errors independent of any model's ability to read the clinical note text",
    ],
    "missing_components": [
        "A discussion of data use agreement, credentialing, and de-identification verification requirements for working with MIMIC-derived clinical text",
        "An analysis or discussion of inherent label noise in ICD billing codes and its impact on both training and evaluation validity",
    ],
    "novelty_score": 0.62,
    "quality_score": 0.79,
}

CONFIDENCE_JSON = {"internal_consistency": 0.82, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to automated ICD code assignment from clinical notes, based on the retrieved abstract." for t in titles]
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
