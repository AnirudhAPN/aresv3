import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer-Based Code Generation for Automated Software Repair"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate how transformer-based code generation models can localize and repair software defects across multiple programming languages.",
        "Compare fine-tuning strategies for adapting pretrained code language models to automated program repair (APR) tasks.",
        "Investigate the impact of context window size and retrieval-augmented prompting on patch correctness.",
        "Assess the generalizability of transformer-based repair models to real-world, previously unseen bug patterns.",
    ],
    "research_questions": [
        "How do transformer architectures compare to traditional template-based and search-based APR techniques in patch correctness?",
        "What fine-tuning or prompting strategies most improve the plausibility and compilability of generated patches?",
        "Does providing surrounding code context and test failure information improve repair accuracy over isolated buggy-line input?",
        "How well do transformer-based repair models generalize across languages and bug categories not seen during training?",
    ],
    "keywords": [
        "automated program repair",
        "transformer code models",
        "neural bug fixing",
        "code language models",
        "patch generation",
    ],
    "research_scope": "This research focuses on transformer-based neural approaches to automated software repair, spanning encoder-decoder and decoder-only architectures (e.g., CodeT5, CodeBERT-based repair models, and large code LLMs) applied to bug localization and patch generation. The scope covers supervised fine-tuning on bug-fix datasets, prompting-based repair with large pretrained models, and evaluation on standard APR benchmarks such as Defects4J and QuixBugs, while excluding purely rule-based or genetic-programming APR techniques except as baselines for comparison.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of pretrained code language models (CodeT5, CodeBERT, StarCoder) fine-tuned specifically for bug-fixing tasks rather than trained from scratch.",
        "Increasing use of large language models with few-shot or zero-shot prompting for patch generation, reducing reliance on large labeled bug-fix corpora.",
        "Shift toward incorporating test execution feedback and program analysis signals alongside raw source code as model input.",
        "Rising interest in multi-language and cross-project generalization benchmarks beyond single-language datasets like Defects4J.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of transformer-based repair models on security-relevant vulnerability classes rather than general functional bugs.",
            "based_on_papers": ["CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparative cost-benefit analysis of fine-tuning versus prompting large models for APR under realistic compute and latency budgets.",
            "based_on_papers": ["Evaluating Large Language Models Trained on Code"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of generated patches to adversarial or subtly misleading test suites that could cause overfitting to weak test signals.",
            "based_on_papers": ["An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop hybrid pipelines combining static analysis-based fault localization with transformer-based patch synthesis to improve precision.",
        "Extend evaluation benchmarks to include multi-file and cross-function repairs rather than single-hunk fixes.",
        "Explore reinforcement learning from compiler and test feedback to fine-tune repair models beyond supervised next-token prediction.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "HuggingFace Datasets", "url": "https://huggingface.co/datasets", "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.", "verified": True}
    ],
    "algorithms": [
        {
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "The self-attention-based encoder-decoder architecture underlying nearly all modern code generation and repair models, enabling parallelized sequence modeling of source code tokens.",
            "verified": True,
        },
        {
            "name": "CodeT5",
            "url": "https://arxiv.org/abs/2109.00859",
            "description": "An identifier-aware unified pretrained encoder-decoder transformer for code understanding and generation, widely used as a base model for fine-tuned automated program repair.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Patch correctness (percentage of generated patches that are semantically equivalent to the developer's ground-truth fix)",
        "Plausibility rate (percentage of generated patches that pass all associated test cases)",
        "Exact match / compile rate (percentage of generated patches that compile without syntax errors)",
        "Bugs fixed at top-k (number of distinct bugs resolved when sampling k candidate patches per bug)",
    ],
    "implementation_steps": [
        "Collect and preprocess a bug-fix corpus (e.g., paired buggy/fixed function pairs) with tokenization aligned to the target code language model's vocabulary.",
        "Select a pretrained transformer code model (e.g., CodeT5 or a decoder-only code LLM) as the base architecture for fine-tuning.",
        "Fine-tune the model on the bug-fix corpus using a sequence-to-sequence objective, optionally augmenting input with surrounding function context and test failure messages.",
        "Generate candidate patches via beam search or sampling for held-out buggy programs from a benchmark such as Defects4J or QuixBugs.",
        "Validate candidate patches by compiling and executing them against the project's existing test suite to filter for plausibility.",
        "Manually or semi-automatically verify a sample of plausible patches for semantic correctness against developer-authored ground-truth fixes.",
    ],
    "unverified_suggestions": [
        "Incorporating abstract syntax tree (AST) structural embeddings alongside token sequences may improve localization accuracy, though this was not directly benchmarked in the retrieved papers.",
        "Using retrieval-augmented generation to surface similar historical bug fixes from the same codebase could improve patch relevance, but supporting evidence in this literature set is limited.",
        "Combining program repair with automated test generation to synthesize regression tests for generated patches may reduce overfitting risk, though this remains largely unexplored in the retrieved set.",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The research plan clearly grounds the study in established transformer architectures with well-documented prior work (Transformer, CodeT5).",
        "The proposal specifies concrete, measurable evaluation metrics (patch correctness, plausibility rate, compile rate) appropriate for the APR domain.",
        "The implementation steps form a coherent, reproducible pipeline from data preprocessing through patch validation.",
        "The identified gaps around security-relevant bugs and adversarial test robustness point to genuinely underexplored and impactful research directions.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small for a topic as active as neural program repair, limiting confidence in how comprehensive the identified gaps and trends actually are.",
        "The methodology does not specify how many bug-fix pairs are needed for effective fine-tuning, leaving open questions about data sufficiency.",
        "There is limited discussion of computational cost and inference latency trade-offs between fine-tuning smaller models versus prompting large general-purpose code LLMs.",
        "The proposal does not address how to handle multi-hunk or multi-file bugs, which are common in real-world repositories but harder for single-function repair models.",
    ],
    "missing_components": [
        "No explicit baseline comparison against non-neural APR techniques (e.g., GenProg, PraPR) to contextualize the improvement offered by transformer-based methods.",
        "No discussion of dataset licensing or contamination risk between pretraining corpora and evaluation benchmarks like Defects4J.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based automated program repair, based on the retrieved abstract." for t in titles]
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
