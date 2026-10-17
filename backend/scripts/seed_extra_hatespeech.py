import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer Models for Multilingual Hate Speech Detection"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based classifiers that detect hate speech across multiple languages, including languages with limited labeled training data",
        "Evaluate multilingual pretrained transformer architectures for their ability to transfer hate speech detection performance from high-resource to low-resource languages",
        "Investigate how cultural and linguistic context-specific slurs, code-switching, and implicit hate speech affect cross-lingual detection accuracy",
        "Assess fairness and false-positive disparities in hate speech detection across different language and dialect communities",
    ],
    "research_questions": [
        "Do multilingual transformer models (e.g., XLM-R) achieve better zero-shot and few-shot cross-lingual transfer for hate speech detection than training separate monolingual models per language?",
        "How does classification accuracy differ between explicit hate speech and implicit or context-dependent hate speech across languages?",
        "What is the impact of code-switching (mixing languages within a single message) on multilingual hate speech classifier accuracy?",
        "Are false-positive and false-negative rates for hate speech detection distributed unevenly across different language, dialect, or cultural communities?",
    ],
    "keywords": [
        "multilingual hate speech detection",
        "cross-lingual transfer learning",
        "offensive language classification",
        "transformer text classification",
        "low-resource language NLP",
    ],
    "research_scope": (
        "This research covers transformer-based methods for detecting hate speech across multiple languages, "
        "including cross-lingual transfer from high-resource to low-resource languages, handling of code-switched "
        "text, and fairness analysis across language and dialect communities. It includes both explicit and implicit "
        "hate speech detection. It excludes general sentiment analysis unrelated to hate speech, content moderation "
        "policy design outside of the classification task itself, and image- or video-based hateful content detection."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of massively multilingual pretrained transformers such as XLM-R and mT5 as the standard backbone for cross-lingual hate speech detection",
        "Increasing focus on zero-shot and few-shot cross-lingual transfer evaluation to assess performance on low-resource languages with minimal labeled data",
        "Rising interest in modeling implicit and context-dependent hate speech, which is harder to detect via lexical/keyword-based signals than explicit slurs",
        "Emerging attention to fairness auditing, examining whether false-positive and false-negative rates differ systematically across language or dialect communities",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of zero-shot cross-lingual transfer versus per-language fine-tuning across a broad, matched set of high- and low-resource languages.",
            "based_on_papers": ["Unsupervised Cross-lingual Representation Learning at Scale"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of code-switched message classification accuracy relative to single-language messages within the same multilingual model.",
            "based_on_papers": ["HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: fairness audits quantifying differential false-positive rates across dialect or cultural subgroups within the same language, rather than only across distinct languages.",
            "based_on_papers": ["HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build broad matched-language benchmarks that directly compare zero-shot cross-lingual transfer against per-language fine-tuning",
        "Develop dedicated code-switched hate speech evaluation sets to isolate this failure mode from single-language performance",
        "Establish dialect- and subgroup-level fairness auditing protocols within, not just across, languages",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "HuggingFace Datasets", "url": "https://huggingface.co/datasets", "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "XLM-R (XLM-RoBERTa)",
            "url": "https://arxiv.org/abs/1911.02116",
            "description": "Massively multilingual transformer pretrained on text from over 100 languages, used as the primary backbone for cross-lingual hate speech classification and transfer experiments.",
            "verified": True,
        },
        {
            "name": "mT5",
            "url": "https://arxiv.org/abs/2010.11934",
            "description": "Multilingual text-to-text transformer used as an alternative architecture for generative-style hate speech classification and comparison against encoder-only XLM-R.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Macro-averaged F1-score per language",
        "Zero-shot and few-shot cross-lingual transfer accuracy",
        "Classification accuracy on code-switched versus single-language messages",
        "False-positive rate disparity across dialect/cultural subgroups",
    ],
    "implementation_steps": [
        "Assemble a multilingual hate speech corpus spanning high-resource and low-resource languages, including a code-switched message subset where available",
        "Fine-tune XLM-R and mT5 on a subset of high-resource languages, then evaluate zero-shot and few-shot transfer to held-out low-resource languages",
        "Fine-tune per-language monolingual baselines where sufficient labeled data exists, for direct comparison against the multilingual transfer setting",
        "Evaluate classification accuracy separately on explicit-slur messages, implicit/context-dependent messages, and code-switched messages",
        "Conduct a fairness audit comparing false-positive and false-negative rates across identifiable dialect or cultural subgroups within each language",
        "Analyze failure cases where cultural context or region-specific terminology causes systematic misclassification",
    ],
    "unverified_suggestions": ["HateXplain dataset", "Multilingual HateCheck", "HASOC multilingual hate speech dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets the practically important zero-shot cross-lingual transfer question rather than only reporting per-language fine-tuned accuracy",
        "Separating explicit, implicit, and code-switched message evaluation gives a much more diagnostic picture of where multilingual models actually fail",
        "The dialect- and subgroup-level fairness audit goes beyond typical cross-language comparisons to address within-language equity concerns",
        "Comparing an encoder-only (XLM-R) and a text-to-text (mT5) architecture provides a broader methodological contribution than a single-model study",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: HuggingFace Datasets is a general-purpose hub, and it is not confirmed that it hosts sufficiently large labeled hate speech corpora for all the low-resource languages the proposal targets, so HateXplain, HateCheck, and HASOC remain unverified suggestions.",
        "The proposal does not address annotator cultural background and its potential influence on ground-truth hate speech labels, which is a known source of label inconsistency in this domain.",
        "There is limited discussion of how the system would handle newly coined slurs or rapidly evolving coded language that would not be present in any static training corpus.",
    ],
    "missing_components": [
        "A discussion of annotator background and cultural context effects on ground-truth label reliability.",
        "A plan for handling novel or rapidly evolving coded hateful language not covered by static training data.",
    ],
    "novelty_score": 0.75,
    "quality_score": 0.80,
}

CONFIDENCE_JSON = {"internal_consistency": 0.88, "specificity": 0.85, "completeness": 0.90}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based multilingual hate speech detection, based on the retrieved abstract." for t in titles]
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
