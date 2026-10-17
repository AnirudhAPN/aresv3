import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Multi-Task Learning for Joint Named Entity Recognition and Relation Extraction"

PLANNER_JSON = {
    "research_objectives": [
        "Design a multi-task neural architecture that jointly performs named entity recognition and relation extraction within a single shared model",
        "Quantify the extent to which joint training on entity and relation labels improves downstream extraction accuracy compared to pipeline approaches that train each task independently",
        "Investigate parameter-sharing and loss-weighting strategies that mitigate negative transfer between the entity recognition and relation extraction subtasks",
        "Evaluate the robustness of joint NER-RE models to error propagation and cascading mistakes relative to traditional two-stage pipelines",
    ],
    "research_questions": [
        "Does joint multi-task learning of named entity recognition and relation extraction outperform independently trained pipeline models on end-to-end triple extraction accuracy?",
        "What shared representation architectures (e.g., shared encoder with task-specific heads vs. table-filling vs. span-based joint decoding) best balance entity and relation performance?",
        "How does the choice of loss weighting between the NER and RE objectives affect negative transfer and overall model convergence?",
        "To what extent does joint modeling reduce error propagation from entity boundary mistakes into downstream relation predictions compared to pipeline systems?",
    ],
    "keywords": [
        "joint entity and relation extraction",
        "multi-task learning NLP",
        "named entity recognition",
        "relation extraction",
        "information extraction pipelines",
    ],
    "research_scope": "This research focuses on multi-task neural architectures that jointly learn named entity recognition (NER) and relation extraction (RE) from unstructured text, spanning shared-encoder multi-head designs, span-based joint decoders, and table-filling formulations. The scope covers supervised joint extraction on standard benchmark corpora such as ACE05, CoNLL04, and SciERC, with attention to loss-weighting and parameter-sharing strategies that reduce negative transfer between the two subtasks. It excludes open information extraction, zero-shot or few-shot relation extraction, and extraction from multimodal or non-English sources; the emphasis remains on architectural design choices and empirical comparison of joint versus pipelined extraction under supervised training regimes.",
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of span-based joint models that enumerate candidate entity spans and score relation compatibility between span pairs in a single forward pass, replacing older sequence-tagging plus classifier pipelines",
        "Increasing use of table-filling and grid-tagging formulations that cast joint NER and RE as a unified word-pair classification problem to natively capture overlapping entities and relations",
        "Wider use of pretrained transformer encoders (e.g., BERT, SciBERT) as shared backbones for both the entity and relation heads, with lightweight task-specific decoders on top",
        "Continued reliance on multi-task loss weighting (fixed or learned, e.g., uncertainty weighting) to balance the entity recognition and relation classification objectives during joint training",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic study of how negative transfer between the NER and RE objectives varies across domains with differing entity density and relation sparsity (e.g., biomedical vs. news text).",
            "based_on_papers": ["A Frustratingly Easy Approach for Joint Entity and Relation Extraction"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: comparison of joint NER-RE models against pipeline baselines under realistic noisy or out-of-domain input, isolating how much of the reported joint-model gain is attributable to reduced error propagation versus representation sharing alone.",
            "based_on_papers": ["Table-Sequence Encoders for Joint Entity and Relation Extraction", "SpERT: Span-based Joint Entity and Relation Extraction with Transformer Pre-training"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: principled methods for automatically adapting the entity-versus-relation loss weighting during training rather than relying on fixed hyperparameter search over static weights.",
            "based_on_papers": ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop domain-stratified benchmarks that make it possible to measure negative transfer between NER and RE objectives as a function of entity density and relation sparsity",
        "Design controlled ablation studies that separate the contribution of shared representations from the contribution of reduced error propagation in joint extraction models",
        "Explore learned or dynamic loss-weighting schemes (e.g., gradient-based task balancing) for the joint NER-RE objective as an alternative to fixed weighting hyperparameters",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "GLUE Benchmark", "url": "https://gluebenchmark.com/", "description": "Suite of NLP understanding tasks used for benchmarking language models.", "verified": True}],
    "algorithms": [
        {
            "name": "BERT",
            "url": "https://arxiv.org/abs/1810.04805",
            "description": "Bidirectional transformer pretraining approach commonly used as the shared encoder backbone for joint NER and relation extraction models, providing contextualized token representations for both task heads.",
            "verified": True,
        },
        {
            "name": "SpERT",
            "url": "https://arxiv.org/abs/1909.07755",
            "description": "Span-based joint entity and relation extraction architecture that enumerates candidate spans from a shared transformer encoder and jointly classifies entity types and pairwise relations in a single model.",
            "verified": True,
        },
    ],
    "evaluation_metrics": ["Entity F1", "Relation F1", "Strict relation triple F1", "Macro-averaged F1 across entity types"],
    "implementation_steps": [
        "Collect and preprocess annotated corpora with aligned entity and relation labels, such as CoNLL04, ACE05, or SciERC",
        "Encode input sentences with a shared pretrained transformer encoder to obtain contextualized token representations",
        "Enumerate or tag candidate entity spans and train an entity classification head on top of the shared encoder",
        "Construct span-pair or word-pair representations for candidate entity pairs and train a relation classification head jointly with the entity head",
        "Tune the multi-task loss weighting between the entity and relation objectives, using either fixed weights or a learned balancing scheme",
        "Evaluate the joint model against independently trained pipeline baselines using entity F1, relation F1, and strict triple extraction F1 on held-out test data",
    ],
    "unverified_suggestions": ["ACE05 relation extraction corpus", "CoNLL04 entity and relation dataset", "SciERC scientific information extraction dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal targets a well-established and practically important information extraction problem with clear, testable research questions comparing joint versus pipeline architectures",
        "The methodology builds on credible, widely adopted architectures (BERT-based shared encoders, span-based joint decoding) rather than proposing an unproven design from scratch",
        "The use of multiple complementary metrics (entity F1, relation F1, strict triple F1) reflects an appropriate awareness that end-to-end extraction quality cannot be judged from entity accuracy alone",
        "The research questions move beyond a general claim that joint learning helps, toward specific mechanisms such as negative transfer and error propagation that can be empirically isolated",
    ],
    "weaknesses": [
        "The proposal does not specify how entity density and relation sparsity will be quantified when comparing domains, leaving the negative-transfer analysis somewhat underspecified",
        "Evidence sufficiency is limited: the gap analysis draws on a small retrieved literature sample, and dataset fit between the proposed benchmarks (CoNLL04, ACE05, SciERC) and the stated cross-domain research questions has not been independently verified",
        "There is limited discussion of computational cost and training time trade-offs introduced by joint multi-task training relative to simpler pipeline baselines",
        "The plan does not address how overlapping or nested entities, which are common in benchmarks like ACE05, will be handled by the chosen span enumeration strategy",
    ],
    "missing_components": [
        "A concrete ablation plan isolating the effect of shared representations from the effect of reduced error propagation on joint model performance",
        "A discussion of statistical significance testing when comparing joint and pipeline model results across multiple benchmark datasets",
    ],
    "novelty_score": 0.67,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to joint multi-task learning of named entity recognition and relation extraction, based on the retrieved abstract." for t in titles]
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
    from config import settings
    import orchestrator as orch

    session_id = memory_service.create_session(TOPIC, source="seed")
    print("session_id:", session_id)

    # .env sets LLM_PROVIDER=local, which makes agents bypass llm_service and
    # call the real local LoRA model directly (services/local_model.py). For
    # seeding synthetic training data we need the mocks below to actually be
    # hit, so force the "api" routing for the duration of this run.
    with patch.object(settings, "llm_provider", "api"), \
         patch.object(llm_service, "generate", side_effect=mock_generate), \
         patch.object(llm_service, "judge", side_effect=mock_judge):
        final_state = orch.run_pipeline(session_id, TOPIC, condition="adaptive")

    print("completed_agents:", final_state.get("completed_agents"))
    print("acs_scores:", json.dumps(final_state.get("acs_scores", {}), indent=2))
    print("flags:", final_state.get("flags"))
    session = memory_service.get_session(session_id)
    print("FINAL STATUS:", session["status"])


if __name__ == "__main__":
    main()
