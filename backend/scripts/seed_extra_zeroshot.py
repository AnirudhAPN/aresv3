import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Zero-Shot Learning for Cross-Lingual Text Classification"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a zero-shot text classification approach that transfers label knowledge from a high-resource source language to unseen target languages without task-specific labeled data in those languages.",
        "Evaluate how multilingual pretrained language models encode label semantics and how well those representations generalize across typologically diverse languages.",
        "Quantify the performance gap between zero-shot cross-lingual transfer and fully supervised in-language baselines across a range of classification tasks.",
        "Identify which linguistic and resource-related factors (script, language family, pretraining corpus size) most strongly predict zero-shot transfer success.",
    ],
    "research_questions": [
        "How effectively do multilingual transformer encoders transfer text classification decision boundaries from English to low-resource target languages with no fine-tuning data?",
        "Does natural language inference-style label formulation (treating class labels as hypotheses) outperform standard softmax classification heads in zero-shot cross-lingual settings?",
        "What is the relationship between a target language's representation in pretraining data and its zero-shot classification accuracy?",
        "Can lightweight adapter modules or prompt-based tuning narrow the zero-shot cross-lingual performance gap without requiring target-language labels?",
    ],
    "keywords": [
        "zero-shot cross-lingual transfer",
        "multilingual text classification",
        "cross-lingual NLI",
        "multilingual language models",
        "low-resource NLP",
    ],
    "research_scope": "This research focuses on zero-shot cross-lingual text classification, where a model trained exclusively on labeled data in a source language (typically English) must classify text in target languages it has never seen labeled examples for. The scope covers multilingual pretrained encoders such as mBERT, XLM-R, and related models, transfer mechanisms including shared subword vocabularies and NLI-based label reformulation, and evaluation across topic classification, sentiment analysis, and natural language inference tasks spanning high-, medium-, and low-resource languages. The work does not cover machine-translation-based pipelines as a primary method, though translate-test baselines are considered for comparison, nor does it address speech or multimodal classification.",
}

GAP_JSON = {
    "current_trends": [
        "Multilingual pretrained encoders such as mBERT and XLM-R are the dominant backbone for zero-shot cross-lingual transfer, relying on shared subword embeddings across languages.",
        "Reformulating classification as a natural language inference task (label-as-hypothesis) is increasingly used to improve zero-shot transfer without target-language fine-tuning.",
        "Performance on zero-shot cross-lingual benchmarks correlates strongly with a language's share of pretraining corpus tokens, disadvantaging low-resource and typologically distant languages.",
        "Parameter-efficient adaptation methods (adapters, prompt tuning) are being explored as alternatives to full fine-tuning for narrowing the cross-lingual transfer gap.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of zero-shot transfer robustness for languages with non-Latin scripts and limited representation in multilingual pretraining corpora.",
            "based_on_papers": ["Unsupervised Cross-lingual Representation Learning at Scale"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: a unified comparison isolating whether gains from NLI-based label reformulation stem from the reformulation itself or from the underlying NLI pretraining data.",
            "based_on_papers": ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: analysis of computational and data efficiency trade-offs between full fine-tuning and adapter-based cross-lingual transfer at scale.",
            "based_on_papers": [
                "Unsupervised Cross-lingual Representation Learning at Scale",
                "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            ],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Extend zero-shot cross-lingual evaluation to a broader and more typologically diverse set of low-resource languages, particularly those underrepresented in pretraining corpora.",
        "Investigate hybrid approaches that combine NLI-based label reformulation with parameter-efficient adapters to improve transfer while controlling compute cost.",
        "Develop diagnostic benchmarks that separate the contribution of shared vocabulary, shared syntax, and shared semantics to zero-shot transfer performance.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "GLUE Benchmark", "url": "https://gluebenchmark.com/", "description": "Suite of NLP understanding tasks used for benchmarking language models.", "verified": True}],
    "algorithms": [
        {
            "name": "BERT",
            "url": "https://arxiv.org/abs/1810.04805",
            "description": "Bidirectional transformer encoder pretrained with masked language modeling; provides the base architecture underlying most multilingual classification approaches.",
            "verified": True,
        },
        {
            "name": "XLM-RoBERTa",
            "url": "https://arxiv.org/abs/1911.02116",
            "description": "Multilingual transformer pretrained on 2.5TB of CommonCrawl data across 100 languages, used as the primary zero-shot cross-lingual transfer backbone in this study.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Macro-averaged F1 score per target language",
        "Zero-shot transfer accuracy relative to source-language upper bound",
        "Cross-lingual transfer gap (source-language accuracy minus zero-shot target-language accuracy)",
        "Area under the accuracy-vs-pretraining-resource curve across target languages",
    ],
    "implementation_steps": [
        "Fine-tune a multilingual encoder (XLM-R) on labeled English classification data only, using standard cross-entropy loss with a linear classification head.",
        "Reformulate the same task as an NLI problem where each class label is converted into a natural language hypothesis, and fine-tune a second model variant on English NLI-style data.",
        "Evaluate both model variants directly on target-language test sets without any target-language fine-tuning or translation.",
        "Stratify target languages by pretraining corpus size and script type to analyze performance variation across resource levels.",
        "Apply lightweight adapter modules on top of the frozen multilingual encoder and repeat zero-shot evaluation to measure efficiency-accuracy trade-offs.",
        "Compare all zero-shot results against a translate-test baseline that machine-translates target-language text into English before classification.",
    ],
    "unverified_suggestions": [
        "Consider incorporating a small amount of target-language unlabeled data via self-training to further close the transfer gap.",
        "Explore whether retrieval-augmented label descriptions improve zero-shot accuracy for extremely low-resource languages.",
        "Investigate calibration of prediction confidence across languages, since zero-shot models are often overconfident on out-of-distribution target languages.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The research questions are precise and directly testable, contrasting NLI-based reformulation against standard classification heads under matched conditions.",
        "The methodology grounds itself in well-established, verifiable multilingual architectures (BERT, XLM-R) rather than speculative or unverifiable methods.",
        "Stratifying evaluation by pretraining resource level and script type gives the study a clear mechanism for explaining, not just measuring, the transfer gap.",
        "Including a translate-test baseline provides a meaningful reference point that grounds the zero-shot results against a widely used alternative pipeline.",
    ],
    "weaknesses": [
        "The literature base of 8 retrieved papers is not sufficient to make strong claims about coverage of the full zero-shot cross-lingual transfer landscape, particularly for adapter-based methods.",
        "The proposal does not specify how many target languages or which typological families will be included, which limits reproducibility of the resource-stratification analysis.",
        "No explicit plan is given for statistical significance testing when comparing transfer gaps across many languages, risking spurious conclusions from noisy low-resource splits.",
        "The connection between adapter efficiency gains and actual zero-shot accuracy improvements is asserted as a future direction but not built into the core experimental design.",
    ],
    "missing_components": [
        "A concrete list of target languages and their resource tiers to be used in evaluation.",
        "A defined compute/parameter budget for comparing adapter-based versus full fine-tuning approaches.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.74,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to zero-shot cross-lingual text classification, based on the retrieved abstract." for t in titles]
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
