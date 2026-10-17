import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Machine Translation for Low-Resource African Languages"

PLANNER_JSON = {
    "research_objectives": [
        "Develop neural machine translation models that achieve usable translation quality for African language pairs with limited parallel corpora",
        "Investigate transfer learning and multilingual pretraining strategies to reduce the amount of labeled data required per language",
        "Evaluate the effectiveness of data augmentation techniques such as back-translation for morphologically rich, low-resource African languages",
        "Assess the practical deployment challenges of NMT systems for African languages, including tokenization of agglutinative morphology and dialectal variation",
    ],
    "research_questions": [
        "How does multilingual pretraining on related African languages affect translation quality compared to training from scratch on a single low-resource pair?",
        "What data augmentation strategies most effectively compensate for the scarcity of parallel text in African language corpora?",
        "How do subword tokenization choices affect translation quality for morphologically complex African languages such as Swahili, Yoruba, and Zulu?",
        "What is the gap in translation quality between high-resource benchmark languages and low-resource African languages under comparable model architectures?",
    ],
    "keywords": [
        "low-resource neural machine translation",
        "African languages NLP",
        "multilingual transfer learning",
        "back-translation data augmentation",
        "subword tokenization morphology",
    ],
    "research_scope": "This research focuses on neural machine translation systems for low-resource African languages, including but not limited to Swahili, Yoruba, Hausa, Zulu, and Amharic, with an emphasis on transfer learning, multilingual pretraining, and data augmentation methods that mitigate scarce parallel corpora. The scope covers text-to-text translation between African languages and high-resource pivot languages such as English and French, and excludes speech-to-speech translation, dialogue systems, and text-to-speech synthesis. Sociolinguistic and policy considerations around language preservation are acknowledged but are not the primary focus; the emphasis remains on model architectures, training strategies, and empirical evaluation of translation quality.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of multilingual pretrained models (e.g., mBART, M2M-100, NLLB) as initialization for low-resource African language pairs rather than training transformer models from scratch",
        "Increasing use of back-translation and other synthetic data generation techniques to expand limited parallel corpora for African languages",
        "Community-driven corpus creation efforts (e.g., Masakhane) that crowdsource parallel data and evaluation benchmarks for previously under-resourced African languages",
        "Growing interest in subword and morphology-aware tokenization schemes tailored to agglutinative and tonal African languages",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of tokenization strategies (BPE vs. morphologically-aware segmentation) across a broad set of African languages with differing morphological typologies.",
            "based_on_papers": ["Massively Multilingual Neural Machine Translation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of NMT robustness to dialectal and orthographic variation within a single African language, which is common given inconsistent standardization.",
            "based_on_papers": ["Transfer Learning for Low-Resource Neural Machine Translation", "Multilingual Denoising Pre-training for Neural Machine Translation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cost-benefit analysis of human-in-the-loop active learning for parallel corpus creation compared to purely automated back-translation pipelines for African languages.",
            "based_on_papers": ["Attention Is All You Need"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design morphology-aware subword tokenizers evaluated specifically on agglutinative and tonal African language families",
        "Build standardized multi-dialect evaluation benchmarks that capture orthographic and dialectal variation within individual African languages",
        "Explore hybrid pipelines combining active learning with back-translation to prioritize human annotation effort on the most informative low-resource sentence pairs",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True}],
    "algorithms": [
        {
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention-based sequence-to-sequence architecture that serves as the standard backbone for modern NMT systems, including low-resource African language translation models.",
            "verified": True,
        },
        {
            "name": "Transfer Learning for Low-Resource NMT",
            "url": "https://arxiv.org/abs/1604.02201",
            "description": "Approach that initializes a low-resource language pair's NMT model with parameters learned from a related high-resource parent language pair, substantially improving translation quality when parallel data is scarce.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["BLEU", "chrF", "TER", "COMET"],
    "implementation_steps": [
        "Collect and clean parallel corpora for the target African language pairs from sources such as Masakhane, JW300, and OPUS",
        "Preprocess text with a morphology-aware subword tokenizer suited to the target language's typology",
        "Pretrain or select a multilingual encoder-decoder model on related high-resource and African language data",
        "Fine-tune the pretrained model on the target low-resource language pair using transfer learning",
        "Apply back-translation using monolingual African language text to augment the parallel training data",
        "Evaluate translation quality on held-out test sets using BLEU, chrF, TER, and COMET, and conduct human evaluation for a subset of outputs",
    ],
    "unverified_suggestions": ["Masakhane MT benchmark corpus", "JW300 parallel corpus", "FLORES-101 evaluation benchmark"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly targets a well-motivated and underserved research area with significant real-world impact for African language communities",
        "The methodology grounds itself in established transformer and transfer learning techniques with a credible, incremental extension to low-resource settings",
        "The use of multiple complementary evaluation metrics (BLEU, chrF, TER, COMET) reflects awareness of the limitations of any single automatic metric",
        "The research questions are specific and testable, moving beyond a generic call for 'better African NLP' toward concrete comparisons of tokenization and transfer strategies",
    ],
    "weaknesses": [
        "The proposal does not fully specify how morphological typology will be quantified or compared across the selected African languages",
        "Evidence sufficiency is limited: the gap analysis is based on a small retrieved literature sample and dataset fit for several proposed African language pairs has not been independently verified against the cited corpora",
        "There is limited discussion of computational resource constraints that are especially acute for research groups based in the languages' countries of origin",
        "The plan does not address how dialectal variation within a language will be handled during evaluation, despite noting it as a gap",
    ],
    "missing_components": [
        "A concrete plan for community and native-speaker involvement in corpus validation and translation quality assessment",
        "A discussion of ethical considerations around data sourcing and consent for community-contributed African language text",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to low-resource neural machine translation for African languages, based on the retrieved abstract." for t in titles]
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
