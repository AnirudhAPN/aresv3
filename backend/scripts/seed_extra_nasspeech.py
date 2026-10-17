import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Architecture Search for Multilingual Speech Translation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop neural architecture search (NAS) methods that automatically discover encoder-decoder architectures optimized for direct speech-to-text translation across multiple languages.",
        "Investigate how search spaces tailored to speech translation, including convolutional subsampling, conformer blocks, and cross-attention layers, affect translation quality and inference latency.",
        "Evaluate the extent to which architectures discovered by NAS transfer across typologically diverse language pairs without requiring per-language re-search.",
        "Assess the computational cost of NAS for multilingual speech translation relative to the downstream gains in BLEU score and inference efficiency on edge and server hardware.",
    ],
    "research_questions": [
        "How do differentiable and evolutionary NAS strategies compare when searching for speech translation architectures across a shared multilingual search space?",
        "What architectural motifs discovered via NAS generalize best across high-resource and low-resource language pairs in multilingual speech translation?",
        "How does architecture search interact with multilingual training regimes, such as language-conditioned adapters or shared encoders, to affect translation quality?",
        "What is the trade-off between search cost, model size, and translation accuracy when applying NAS to on-device multilingual speech translation systems?",
    ],
    "keywords": [
        "neural architecture search",
        "multilingual speech translation",
        "speech-to-text translation",
        "conformer architectures",
        "differentiable NAS",
    ],
    "research_scope": (
        "This research focuses on neural architecture search techniques applied to end-to-end multilingual speech "
        "translation systems, covering direct speech-to-text translation models that map source-language audio "
        "to target-language text without an intermediate transcription step. The scope includes differentiable "
        "NAS, evolutionary search, and reinforcement-learning-based search strategies applied to encoder-decoder "
        "and conformer-based architectures, as well as search spaces incorporating convolutional subsampling, "
        "attention mechanisms, and language-conditioned adapter modules for multilingual generalization. Excluded "
        "from this scope are cascaded pipelines that rely on separate automatic speech recognition and machine "
        "translation modules, manually designed architectures without any automated search component, and "
        "monolingual speech translation systems restricted to a single language pair, which are considered out "
        "of scope for this review."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of conformer-based encoders as the dominant backbone within NAS search spaces for speech translation, combining convolution and self-attention for local and global context.",
        "Increasing use of differentiable NAS methods, such as DARTS-style continuous relaxation, to reduce the computational cost of searching large speech translation architecture spaces.",
        "Rising interest in multilingual search spaces that incorporate language-conditioned adapters or shared subnetworks to enable architecture reuse across many language pairs.",
        "Emerging work on hardware-aware NAS objectives that jointly optimize translation quality and inference latency for deployment on edge devices.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarks for evaluating how well NAS-discovered speech translation architectures transfer across typologically distant language pairs, such as agglutinative versus isolating languages.",
            "based_on_papers": ["Neural Architecture Search: A Survey"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of the search cost and carbon footprint of differentiable versus evolutionary NAS strategies specifically for multilingual speech translation, as opposed to image classification or general NLP tasks.",
            "based_on_papers": ["DARTS: Differentiable Architecture Search"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: end-to-end evaluation of NAS-discovered architectures under real-world streaming and low-latency constraints, rather than only offline BLEU score comparisons on curated test sets.",
            "based_on_papers": ["Neural Architecture Search: A Survey", "DARTS: Differentiable Architecture Search"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop typology-aware benchmarks that measure cross-lingual transferability of NAS-discovered speech translation architectures.",
        "Explore hardware-aware, carbon-efficient NAS objectives that jointly optimize translation quality, latency, and search cost for multilingual deployment.",
        "Design streaming-oriented evaluation protocols that assess NAS-discovered architectures under realistic low-latency, on-device translation constraints.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "HuggingFace Datasets",
            "url": "https://huggingface.co/datasets",
            "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "An attention-based sequence-to-sequence architecture that forms the backbone of most modern speech translation decoders and many NAS search space building blocks.",
            "verified": True,
        },
        {
            "name": "DARTS (Differentiable Architecture Search)",
            "url": "https://arxiv.org/abs/1806.09055",
            "description": "A gradient-based neural architecture search method that relaxes the discrete architecture search space into a continuous one, enabling efficient joint optimization of architecture and weights via gradient descent.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "BLEU score (translation quality)",
        "Word error rate on intermediate transcription proxies",
        "Inference latency per utterance",
        "Search cost in GPU-hours",
    ],
    "implementation_steps": [
        "Assemble a multilingual speech translation corpus spanning multiple source languages and a shared or per-language target language",
        "Define a NAS search space covering convolutional subsampling, conformer encoder blocks, and cross-attention decoder layers",
        "Implement a differentiable NAS procedure, such as DARTS-style continuous relaxation, to jointly optimize architecture weights and model parameters",
        "Run architecture search under a fixed compute budget and select the top-performing candidate architectures via validation BLEU score",
        "Retrain the selected architectures from scratch on the full multilingual dataset and evaluate cross-lingual transfer to held-out language pairs",
        "Benchmark inference latency and model size on target hardware to assess the quality-efficiency trade-off relative to manually designed baselines",
    ],
    "unverified_suggestions": [
        "CoVoST 2 multilingual speech translation corpus",
        "MuST-C speech translation corpus",
        "Common Voice multilingual speech dataset",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly motivates applying neural architecture search to multilingual speech translation with well-grounded algorithmic references such as DARTS and the Transformer.",
        "The research scope precisely bounds the investigation to direct, end-to-end speech translation architectures, explicitly excluding cascaded ASR-plus-MT pipelines and manually designed models.",
        "The methodology lays out a coherent pipeline from search space definition through differentiable search, retraining, and cross-lingual transfer evaluation.",
        "The identified gaps around typological transferability and hardware-aware search cost are relevant and point toward actionable future work.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small, so conclusions about broader trends in NAS for speech translation should be treated as preliminary.",
        "The proposal does not fully address dataset fit, since only the generic HuggingFace Datasets hub is verified while the topic-specific corpora, such as CoVoST 2 and MuST-C, remain unverified suggestions.",
        "The trade-off between search cost and translation quality is identified as an objective but not translated into concrete profiling steps or a fixed compute budget in the implementation plan.",
        "The proposal does not specify how cross-lingual transfer will be validated statistically beyond BLEU score comparisons across language pairs.",
    ],
    "missing_components": [
        "A concrete protocol for selecting and stratifying typologically diverse language pairs to test cross-lingual architecture transfer.",
        "An ablation plan isolating the contribution of individual search space components, such as conformer blocks versus adapter modules, to translation quality gains.",
    ],
    "novelty_score": 0.74,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to neural architecture search for multilingual speech translation, based on the retrieved abstract." for t in titles]
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
