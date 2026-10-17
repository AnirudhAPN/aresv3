import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Machine Translation for Code-Switched Multilingual Text"

PLANNER_JSON = {
    "research_objectives": [
        "Develop neural machine translation architectures that robustly handle intra-sentential code-switching between two or more languages",
        "Quantify how code-switching frequency and switch-point density affect translation quality across language pairs",
        "Design training strategies that mitigate the scarcity of parallel code-switched corpora",
        "Evaluate whether multilingual pretrained models transfer effectively to code-switched translation without task-specific fine-tuning",
    ],
    "research_questions": [
        "How do transformer-based NMT models degrade in translation quality as the rate of code-switching within a sentence increases?",
        "Can synthetic code-switched data generation improve translation performance over training on monolingual or naturally occurring switched corpora alone?",
        "What tokenization and subword segmentation strategies best preserve morphological boundaries at language switch points?",
        "Does joint multilingual pretraining (e.g., mBART, mT5) provide a better initialization for code-switched NMT than bilingual pretraining?",
    ],
    "keywords": [
        "code-switched machine translation",
        "multilingual NMT",
        "synthetic code-switching generation",
        "low-resource language pairs",
        "subword tokenization for mixed-language text",
    ],
    "research_scope": "This research focuses on neural machine translation systems for code-switched text, i.e., text that alternates between two or more languages within a single sentence or discourse segment, as commonly observed in multilingual communities and social media (e.g., Hindi-English, Spanish-English). The scope covers model architectures (transformer encoder-decoder variants, multilingual pretrained models), data strategies (synthetic code-switch generation, transliteration handling, back-translation), and evaluation methodology (BLEU/chrF under varying switch-point density) for translating code-switched input into a single target language or normalized monolingual output. It excludes speech-based code-switching (ASR) and focuses purely on text-to-text translation.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of large multilingual pretrained encoder-decoders (mBART, mT5, NLLB) as strong initializations for code-switched translation tasks",
        "Growing reliance on synthetically generated code-switched training data via linguistic constraint-based substitution (e.g., Equivalence Constraint, Matrix Language Frame) due to scarce naturally annotated parallel corpora",
        "Shift toward subword and byte-level tokenization (SentencePiece, BPE) to handle script mixing and out-of-vocabulary switch points",
        "Emerging use of large language models with few-shot prompting for code-switched translation as an alternative to dedicated fine-tuned NMT systems",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of translation quality as a continuous function of switch-point density rather than coarse code-switched vs. monolingual comparisons",
            "based_on_papers": ["Neural Machine Translation for Code-Switched Text"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of code-switched NMT to unseen language pairs or unseen script combinations not present in synthetic training data",
            "based_on_papers": ["A Survey of Code-switched Speech and Language Processing"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized human evaluation protocols for code-switched translation output, as automatic metrics like BLEU are known to correlate poorly with fluency judgments on mixed-language text",
            "based_on_papers": ["Overview for the First Shared Task on Language Identification in Code-Switched Data"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop switch-point-aware evaluation metrics that weight translation errors by their proximity to language boundaries",
        "Curate and release larger naturally occurring code-switched parallel corpora across typologically diverse language pairs beyond Hindi-English and Spanish-English",
        "Investigate parameter-efficient adaptation (LoRA, adapters) of large multilingual models specifically for low-resource code-switched pairs",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "HuggingFace Datasets", "url": "https://huggingface.co/datasets", "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.", "verified": True}],
    "algorithms": [
        {"name": "Transformer", "url": "https://arxiv.org/abs/1706.03762", "description": "Encoder-decoder architecture with self-attention, used as the backbone for sequence-to-sequence NMT models including code-switched translation systems.", "verified": True},
        {"name": "mBART (Multilingual Denoising Pretraining)", "url": "https://arxiv.org/abs/2001.08210", "description": "Sequence-to-sequence denoising autoencoder pretrained on monolingual corpora across many languages, commonly used to initialize code-switched NMT fine-tuning.", "verified": True},
    ],
    "evaluation_metrics": ["BLEU", "chrF", "TER", "COMET"],
    "implementation_steps": [
        "Collect and clean naturally occurring code-switched text from social media and conversational corpora, aligning it with reference translations where available",
        "Generate synthetic code-switched training pairs using linguistic constraint models (Equivalence Constraint or Matrix Language Frame theory) applied to parallel monolingual corpora",
        "Train a SentencePiece subword tokenizer jointly on the mixed-language vocabulary to minimize fragmentation at switch points",
        "Fine-tune a multilingual pretrained encoder-decoder (e.g., mBART) on the combined natural and synthetic code-switched parallel data",
        "Evaluate translation quality stratified by switch-point density buckets (low, medium, high) using BLEU, chrF, and COMET",
        "Conduct human evaluation of fluency and adequacy on a held-out sample to validate automatic metric correlations",
    ],
    "unverified_suggestions": [
        "Explore contrastive pretraining objectives that explicitly align embeddings of translation-equivalent tokens across the switched languages",
        "Investigate whether curriculum learning that gradually increases switch-point density during training improves final model robustness",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research questions that isolate switch-point density as a measurable, testable variable",
        "Methodology combines both natural and synthetic code-switched data, addressing the well-known data scarcity problem",
        "Evaluation plan uses multiple complementary metrics (BLEU, chrF, COMET) rather than relying on a single automatic score",
        "Proposal explicitly plans human evaluation to validate automatic metrics, which is important given known metric weaknesses on mixed-language text",
    ],
    "weaknesses": [
        "The evidence base draws on a limited set of retrieved papers, so claims about current trends may not fully capture recent large-language-model-based approaches to code-switched translation",
        "No concrete plan for handling more than two languages in a single code-switched sentence, which is common in some multilingual communities",
        "The proposal does not specify how synthetic code-switch generation quality itself will be validated before being used for training",
        "Dataset fit is uncertain: the listed dataset source (HuggingFace Datasets) is generic and the proposal does not confirm availability of annotated code-switched parallel corpora with sufficient scale for the target language pairs",
    ],
    "missing_components": [
        "A concrete baseline comparison against zero-shot or few-shot prompting of a general-purpose large language model",
        "An error analysis plan that categorizes failure modes by linguistic phenomenon (e.g., morphological blending, script switching, borrowed lexical items)",
    ],
    "novelty_score": 0.7,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to neural translation of code-switched multilingual text, based on the retrieved abstract." for t in titles]
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
