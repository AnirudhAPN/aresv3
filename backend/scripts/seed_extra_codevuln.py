import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer Models for Code Vulnerability Detection in Software Repositories"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based models pretrained on large source code corpora to detect security vulnerabilities at function-level granularity in software repositories",
        "Evaluate code-pretrained transformer architectures (e.g., CodeBERT, GraphCodeBERT) against graph neural network and traditional static analysis baselines for vulnerability classification",
        "Investigate the use of data-flow and control-flow information alongside token sequences to improve vulnerability localization beyond function-level binary classification",
        "Assess cross-project generalization of transformer vulnerability detectors when applied to repositories and vulnerability types not represented in the training corpus",
    ],
    "research_questions": [
        "Do transformer models pretrained on large code corpora outperform graph neural network and traditional static analysis tools for function-level vulnerability classification?",
        "Does incorporating explicit data-flow information (as in GraphCodeBERT) improve vulnerability detection accuracy compared to token-sequence-only pretraining (as in CodeBERT)?",
        "How well do transformer vulnerability detectors generalize to unseen projects and vulnerability categories not represented in the fine-tuning dataset?",
        "Can attention weights or learned representations be used to localize the specific lines within a flagged function most likely responsible for the vulnerability?",
    ],
    "keywords": [
        "code vulnerability detection transformer",
        "CodeBERT source code pretraining",
        "static analysis deep learning",
        "software vulnerability classification",
        "graph neural network code representation",
    ],
    "research_scope": (
        "This research focuses on transformer-based models for detecting and localizing security vulnerabilities in "
        "source code at function-level granularity, covering code-pretrained transformer architectures, "
        "incorporation of data-flow and control-flow structure, and comparison against graph neural network and "
        "traditional static analysis baselines. The scope includes cross-project generalization evaluation. It "
        "excludes dynamic and fuzzing-based vulnerability discovery, binary-level reverse engineering, and automated "
        "exploit generation, which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of large-scale pretrained code language models such as CodeBERT and GraphCodeBERT, fine-tuned for downstream vulnerability classification tasks",
        "Increasing use of graph neural networks over code property graphs (combining abstract syntax tree, control-flow, and data-flow edges) as an alternative or complementary representation to token sequences",
        "Continued reliance on a small number of curated vulnerability datasets (e.g., Devign, Big-Vul) due to the cost and difficulty of reliably labeling real-world vulnerable code",
        "Rising interest in explainability techniques to localize vulnerable lines within a flagged function, moving beyond coarse function-level binary classification",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of cross-project generalization when a transformer vulnerability detector is fine-tuned on one set of repositories and tested on entirely different, unseen projects.",
            "based_on_papers": ["CodeBERT: A Pre-Trained Model for Programming and Natural Languages"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct, matched-data comparison between token-sequence-only pretraining (CodeBERT) and data-flow-augmented pretraining (GraphCodeBERT) specifically for the vulnerability detection task, as opposed to general code understanding benchmarks.",
            "based_on_papers": ["GraphCodeBERT: Pre-training Code Representations with Data Flow"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative validation of attention-based or gradient-based line-level localization against ground-truth vulnerable line annotations, rather than function-level classification accuracy alone.",
            "based_on_papers": ["Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build standardized cross-project evaluation splits that explicitly test generalization to repositories and vulnerability types absent from training",
        "Design controlled ablations that isolate the contribution of data-flow information from pretraining corpus size and architecture differences",
        "Develop and validate line-level localization faithfulness metrics using datasets with ground-truth vulnerable line annotations such as Big-Vul",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "CodeBERT",
            "url": "https://arxiv.org/abs/2002.08155",
            "description": "Transformer model pretrained on paired natural language and source code using masked language modeling, used as the token-sequence-based baseline for vulnerability classification.",
            "verified": True,
        },
        {
            "name": "GraphCodeBERT",
            "url": "https://arxiv.org/abs/2009.08366",
            "description": "Transformer model that extends CodeBERT pretraining with explicit data-flow graph information, used to test whether structural code semantics improve vulnerability detection over token sequences alone.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Precision, Recall, and F1-score for function-level vulnerability classification",
        "Area under the ROC curve (AUROC)",
        "Cross-project generalization accuracy on held-out repositories",
        "Line-level localization accuracy against ground-truth vulnerable lines",
    ],
    "implementation_steps": [
        "Assemble a labeled function-level vulnerability dataset spanning multiple open-source projects and vulnerability categories (e.g., buffer overflow, SQL injection, use-after-free)",
        "Construct data-flow and control-flow graph representations for each function to support GraphCodeBERT and graph neural network baselines",
        "Fine-tune CodeBERT and GraphCodeBERT on the labeled dataset for binary and multi-class vulnerability classification",
        "Train a graph neural network baseline over code property graphs for comparison against the transformer-based models",
        "Evaluate cross-project generalization by holding out entire repositories from training and testing on their functions",
        "Apply attention-based or gradient-based attribution to flagged functions and validate line-level localization against ground-truth vulnerable line annotations where available",
    ],
    "unverified_suggestions": [
        "Devign vulnerability detection dataset",
        "Big-Vul vulnerability dataset",
        "CVEfixes vulnerability-fix dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal targets cross-project generalization directly, which is a realistic deployment requirement often glossed over in vulnerability detection studies that use random function-level splits",
        "Comparing CodeBERT against GraphCodeBERT under matched conditions is a well-motivated, isolatable research question rather than an arbitrary architecture choice",
        "Extending evaluation beyond function-level classification to line-level localization addresses a genuinely more useful capability for security engineers triaging flagged code",
        "Including a graph neural network baseline alongside transformer models gives a reasonably comprehensive comparison across the two dominant code representation paradigms",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not address how vulnerability label quality and consistency across the assembled multi-project dataset will be verified, given known noise in CVE-to-commit mapping used to build such datasets",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host source code vulnerability datasets, so the project depends on the unverified Devign, Big-Vul, and CVEfixes suggestions, which vary in labeling methodology and language coverage",
        "The plan does not address a genuinely important methodological consideration for this domain: severe class imbalance between vulnerable and non-vulnerable functions in real repositories is not discussed, despite materially affecting precision/recall trade-offs",
    ],
    "missing_components": [
        "A plan for handling class imbalance between vulnerable and non-vulnerable functions during training and evaluation",
        "A discussion of how differences in CVE-to-commit labeling methodology across the assembled datasets could introduce systematic label noise affecting cross-project results",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.77,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer models for code vulnerability detection, based on the retrieved abstract." for t in titles]
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
