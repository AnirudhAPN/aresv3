import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Machine Translation for Sign Language Gloss to Spoken Text"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a neural machine translation model that converts sign language gloss sequences into fluent, grammatically correct spoken-language text",
        "Evaluate transformer-based sequence-to-sequence architectures against RNN/LSTM-based baselines for gloss-to-text translation quality",
        "Investigate low-resource NMT techniques, including back-translation and fine-tuning of pretrained multilingual language models, given the scarcity of parallel gloss-text corpora",
        "Assess translation quality with respect to structural divergence between sign language gloss ordering and target spoken-language syntax",
    ],
    "research_questions": [
        "Can transformer-based NMT models trained on limited parallel gloss-text corpora produce grammatically fluent spoken-language translations?",
        "How much does back-translation and synthetic gloss augmentation improve gloss-to-text translation quality under low-resource conditions?",
        "How does fine-tuning a pretrained multilingual decoder (e.g., mBART) compare to training a transformer from scratch for gloss-to-text translation?",
        "What translation error types are most strongly associated with structural divergence between gloss ordering and target spoken-language grammar?",
    ],
    "keywords": [
        "sign language gloss translation",
        "neural machine translation sign language",
        "gloss to text transformer",
        "sign language translation low-resource NMT",
        "RWTH PHOENIX Weather sign language",
    ],
    "research_scope": (
        "This research focuses on the text-to-text translation stage of sign language translation pipelines, "
        "specifically converting gloss sequences (already extracted from sign video, or human-annotated) into fluent "
        "spoken-language sentences using neural machine translation. It covers transformer-based sequence-to-sequence "
        "modeling, low-resource NMT techniques such as back-translation and pretrained decoder fine-tuning, and "
        "error analysis of structural mismatches between gloss and spoken-language grammar. The scope excludes the "
        "upstream video-to-gloss recognition task (continuous sign language recognition from video) and does not "
        "cover sign language generation (spoken text to sign video/avatar), which are treated as separate research "
        "problems outside this work."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Transformer-based sequence-to-sequence architectures becoming the dominant approach for gloss-to-text translation, building on general-purpose NMT advances",
        "Use of pretrained multilingual language models (mBART, T5, BERT-based decoders) fine-tuned for low-resource gloss translation",
        "Growing use of back-translation and synthetic gloss generation to augment scarce parallel gloss-text corpora",
        "Joint end-to-end modeling of sign video-to-gloss recognition and gloss-to-text translation within a single pipeline",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic error analysis that distinguishes translation errors caused by upstream gloss recognition inaccuracy from errors caused by gloss-to-text NMT structural mismatch.",
            "based_on_papers": ["Neural Sign Language Translation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: limited exploration of multilingual pretrained decoder fine-tuning (e.g., mBART) specifically tailored to gloss-to-text translation, as opposed to general low-resource spoken-language machine translation.",
            "based_on_papers": ["Attention Is All You Need"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of gloss-to-text translation quality across sign language / spoken language pairs beyond the dominant German Sign Language-to-English benchmark setting.",
            "based_on_papers": ["Neural Sign Language Translation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop diagnostic evaluation protocols that separate gloss-recognition error propagation from translation-stage structural errors",
        "Systematically benchmark pretrained multilingual decoder fine-tuning strategies specifically for gloss-to-text translation across multiple sign language corpora",
        "Extend gloss-to-text NMT evaluation to underrepresented sign language / spoken language pairs beyond German Sign Language-English",
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
            "name": "Transformer (sequence-to-sequence)",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention-based encoder-decoder architecture used as the core NMT model for translating gloss token sequences into spoken-language sentences.",
            "verified": True,
        },
        {
            "name": "mBART",
            "url": "https://arxiv.org/abs/2001.08210",
            "description": "Pretrained multilingual sequence-to-sequence denoising autoencoder, fine-tuned as the decoder for low-resource gloss-to-text translation to leverage broad language modeling knowledge.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "BLEU score",
        "METEOR",
        "ROUGE-L",
        "Word Error Rate (WER)",
    ],
    "implementation_steps": [
        "Collect and align parallel gloss-text corpora (e.g., RWTH-PHOENIX-Weather 2014T) into a standardized training format",
        "Tokenize gloss sequences while preserving sign-specific token ordering and handling non-manual markers",
        "Train a transformer sequence-to-sequence baseline model on the parallel gloss-text pairs",
        "Fine-tune a pretrained multilingual decoder (mBART) on the same gloss-text pairs for comparison",
        "Apply back-translation and synthetic gloss augmentation to expand effective training data under low-resource conditions",
        "Evaluate translation quality using BLEU, METEOR, ROUGE-L, and WER, and conduct qualitative error-type analysis on a held-out test set",
    ],
    "unverified_suggestions": ["RWTH-PHOENIX-Weather 2014T", "How2Sign dataset", "ASLG-PC12 corpus"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal precisely scopes the translation stage separately from upstream gloss recognition, avoiding conflation of two distinct research problems",
        "Comparing a from-scratch transformer against a fine-tuned pretrained multilingual decoder gives a clear, well-motivated ablation for the low-resource setting",
        "The augmentation strategy (back-translation, synthetic gloss generation) is directly matched to the stated low-resource data constraint",
        "The evaluation plan combines automatic metrics (BLEU, METEOR, ROUGE-L, WER) with qualitative structural error analysis, giving a fuller picture than automatic metrics alone",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how synthetic gloss sequences for back-translation will be generated or validated for linguistic plausibility, which risks introducing noisy training signal",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host parallel gloss-text corpora, so the project depends entirely on the unverified, domain-specific dataset suggestions",
        "The plan does not address how translation quality will be validated by native signers or Deaf community reviewers, which is important given known limitations of automatic MT metrics for capturing sign language-specific meaning fidelity",
    ],
    "missing_components": [
        "A human evaluation protocol involving Deaf or hard-of-hearing reviewers to assess translation fidelity beyond automatic metrics",
        "An explicit plan for handling non-manual grammatical markers (facial expression, mouth morphemes) that gloss transcription alone may not fully capture",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.71,
}

CONFIDENCE_JSON = {"internal_consistency": 0.78, "specificity": 0.76, "completeness": 0.81}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to neural machine translation from sign language gloss to spoken text, based on the retrieved abstract." for t in titles]
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
