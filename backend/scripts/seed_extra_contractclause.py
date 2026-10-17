import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer-Based Models for Legal Contract Clause Classification"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based classifiers that assign standardized clause-type labels (e.g., indemnification, termination, limitation of liability) to segments of legal contracts",
        "Evaluate long-document transformer architectures for their ability to classify clauses accurately within full-length contracts rather than isolated clause snippets",
        "Investigate how domain-adaptive pretraining on legal corpora affects clause classification accuracy compared to general-purpose pretrained transformers",
        "Assess model performance on rare or ambiguous clause categories that are underrepresented in existing labeled contract datasets",
    ],
    "research_questions": [
        "Do transformer models pretrained on legal text (e.g., Legal-BERT) outperform general-purpose pretrained transformers for clause-type classification accuracy?",
        "How does classification accuracy change when clauses are classified in the context of the full surrounding contract versus as isolated text snippets?",
        "What is the impact of long-context architectures on clause boundary detection and classification accuracy in lengthy, multi-section contracts?",
        "Which data augmentation or few-shot techniques most improve classification accuracy for rare clause categories with limited labeled examples?",
    ],
    "keywords": [
        "contract clause classification",
        "legal NLP",
        "transformer text classification",
        "long document classification",
        "legal document understanding",
    ],
    "research_scope": (
        "This research covers transformer-based methods for classifying clauses within legal contracts into "
        "standardized categories such as indemnification, termination, confidentiality, and limitation of liability, "
        "including both isolated-clause and full-document classification settings. It includes domain-adaptive "
        "pretraining, long-context architectures, and techniques for handling rare clause categories. It excludes "
        "contract drafting or generation, legal question-answering unrelated to clause labeling, and general-purpose "
        "document summarization tasks."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of legal-domain-pretrained transformers (e.g., Legal-BERT) as the standard starting point for contract clause classification tasks",
        "Growing adoption of long-context transformer architectures to classify clauses within their full surrounding contract rather than as isolated snippets",
        "Rising interest in few-shot and data augmentation techniques to address the long-tailed distribution of clause categories in labeled contract corpora",
        "Emerging use of hierarchical or two-stage pipelines that first segment contracts into clause boundaries before applying classification",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of isolated-clause versus full-document classification accuracy using the same underlying transformer architecture and training data.",
            "based_on_papers": ["CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of few-shot or data augmentation strategies specifically for rare clause categories in contract classification, as most benchmarks report only aggregate accuracy dominated by common clause types.",
            "based_on_papers": ["LEDGAR: A Large-Scale Multi-Label Corpus for Text Classification of Legal Provisions in Contracts"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: analysis of clause boundary detection errors and their downstream effect on classification accuracy in automatically segmented (non-gold-boundary) contracts.",
            "based_on_papers": ["CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Construct matched benchmarks that isolate the effect of full-document context versus isolated-clause input on classification accuracy",
        "Develop and evaluate few-shot and augmentation techniques targeted specifically at long-tailed, rare clause categories",
        "Study the propagation of clause-segmentation errors into downstream classification accuracy in fully automated (non-gold-boundary) pipelines",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "HuggingFace Datasets", "url": "https://huggingface.co/datasets", "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "BERT",
            "url": "https://arxiv.org/abs/1810.04805",
            "description": "Bidirectional transformer encoder used as the baseline architecture for contract clause classification, fine-tuned on labeled clause data.",
            "verified": True,
        },
        {
            "name": "Longformer",
            "url": "https://arxiv.org/abs/2004.05150",
            "description": "Long-context transformer with sparse attention, used to classify clauses using their full surrounding contract context rather than truncated snippets.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Macro-averaged F1-score across clause categories",
        "Per-category precision and recall, reported separately for rare clause types",
        "Clause boundary detection accuracy (for automated segmentation pipelines)",
        "Multi-label exact-match accuracy for clauses with overlapping category labels",
    ],
    "implementation_steps": [
        "Assemble a labeled contract corpus with clause-level category annotations, preserving full document context for each clause",
        "Fine-tune BERT on isolated clause snippets as a baseline, and fine-tune Longformer on full-document input to compare context effects",
        "Apply domain-adaptive pretraining using an available legal-text corpus prior to task-specific fine-tuning",
        "Implement data augmentation or few-shot techniques (e.g., back-translation, synonym substitution, prototype-based few-shot classifiers) targeted at rare clause categories",
        "Evaluate a two-stage pipeline that first predicts clause boundaries and then classifies each segment, comparing against gold-boundary classification accuracy",
        "Conduct error analysis on rare-category misclassifications and on documents where automated boundary detection diverges from gold annotations",
    ],
    "unverified_suggestions": ["CUAD (Contract Understanding Atticus Dataset)", "LEDGAR clause classification dataset", "ContractNLI dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal isolates a specific, testable methodological question (full-document versus isolated-clause context) rather than simply benchmarking another transformer variant",
        "Targeting rare clause categories with dedicated few-shot and augmentation techniques addresses a realistic pain point in legal NLP given long-tailed clause distributions",
        "Including automated clause-boundary detection as part of the pipeline, rather than assuming gold boundaries, reflects a more deployment-realistic evaluation setting",
        "The evaluation plan appropriately separates aggregate and per-category metrics, avoiding the common pitfall of common-clause performance masking rare-clause failures",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: HuggingFace Datasets is a general-purpose hub, and it is not confirmed that it hosts fully annotated, full-document-context contract corpora comparable to CUAD or LEDGAR, so those remain unverified suggestions.",
        "The proposal does not address how confidentiality or redaction requirements for real-world contracts (many of which are proprietary) would be handled during data collection.",
        "There is limited discussion of jurisdictional and contract-type variation (e.g., NDAs versus M&A agreements), which can shift clause definitions and boundaries in ways that may affect model transfer.",
    ],
    "missing_components": [
        "A data governance and redaction protocol for handling confidential or proprietary contract text.",
        "A discussion of cross-jurisdiction and cross-contract-type generalization of the trained clause classifiers.",
    ],
    "novelty_score": 0.65,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.80, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based classification of legal contract clauses, based on the retrieved abstract." for t in titles]
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
