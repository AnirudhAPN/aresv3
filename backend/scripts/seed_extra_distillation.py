import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Knowledge Distillation for Efficient On-Device Natural Language Processing"

PLANNER_JSON = {
    "research_objectives": [
        "Develop knowledge distillation techniques that compress large pretrained language models into compact student models suitable for on-device inference.",
        "Quantify the trade-offs between model size, latency, and downstream task accuracy for distilled NLP models running on mobile and embedded hardware.",
        "Investigate task-agnostic versus task-specific distillation strategies for maintaining generalization under aggressive compression.",
        "Explore combining knowledge distillation with complementary compression methods such as quantization and pruning for further on-device efficiency gains.",
    ],
    "research_questions": [
        "How much can transformer-based language models be compressed via knowledge distillation while retaining acceptable accuracy on standard NLU benchmarks?",
        "What distillation objectives (soft-label matching, hidden-state alignment, attention transfer) most effectively preserve linguistic knowledge in small student models?",
        "How do distilled models perform under real-world on-device constraints such as limited memory, battery, and CPU-only inference?",
        "Can distillation be combined with quantization and pruning without compounding accuracy degradation?",
    ],
    "keywords": [
        "knowledge distillation",
        "on-device NLP",
        "model compression",
        "efficient transformers",
        "student-teacher learning",
    ],
    "research_scope": "This research focuses on knowledge distillation methods applied to transformer-based natural language processing models with the explicit goal of enabling efficient inference on resource-constrained devices such as smartphones, wearables, and embedded systems. It covers distillation objectives (logit matching, intermediate representation transfer, attention-based transfer), student architecture design, and evaluation on standard NLU and NLG benchmarks under latency and memory budgets representative of mobile hardware. The scope excludes distillation for computer vision or speech models, cloud-only serving optimizations that do not target edge deployment, and hardware-specific accelerator co-design that falls outside software-level model compression.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of task-agnostic distillation (e.g., DistilBERT-style pretraining distillation) followed by task-specific fine-tuning of the compact student.",
        "Growing use of intermediate-layer and attention-map distillation losses in addition to soft-label KL divergence to transfer richer teacher knowledge.",
        "Increasing combination of distillation with quantization and structured pruning to reach sub-10MB model footprints for mobile deployment.",
        "Emergence of dynamic and multi-teacher distillation setups where student models are trained against ensembles or progressively larger teachers.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of distilled NLP models under real battery and thermal constraints on actual mobile hardware rather than simulated latency benchmarks.",
            "based_on_papers": ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of distilled student models to distribution shift and noisy on-device input (e.g., typos, code-switching) commonly seen in real user text.",
            "based_on_papers": ["TinyBERT: Distilling BERT for Natural Language Understanding"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: principled methods for jointly optimizing distillation temperature, student depth/width, and quantization bit-width as a unified search rather than sequential ad hoc pipelines.",
            "based_on_papers": ["MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design of hardware-aware distillation loss functions that directly optimize for on-device latency and energy consumption alongside accuracy.",
        "Development of continual or federated distillation approaches that let on-device models adapt to local user data without full retraining.",
        "Standardized benchmarks that jointly report accuracy, memory footprint, latency, and energy use across diverse mobile and embedded chipsets.",
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
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "The self-attention based architecture underlying modern large NLP teacher models used as the source of knowledge for distillation.",
            "verified": True,
        },
        {
            "name": "DistilBERT",
            "url": "https://arxiv.org/abs/1910.01108",
            "description": "A task-agnostic distillation method that trains a smaller student transformer to match the output distribution and hidden representations of a larger BERT teacher, reducing model size by roughly 40% while retaining most of its language understanding performance.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "GLUE benchmark accuracy/F1",
        "inference latency (ms) on target device",
        "model size (MB) / parameter count",
        "energy consumption per inference (mJ)",
    ],
    "implementation_steps": [
        "Select and fine-tune a large teacher language model on the target NLP tasks to serve as the knowledge source",
        "Design a compact student architecture with reduced depth, width, or embedding dimensionality suited for on-device constraints",
        "Define the distillation loss combining soft-label KL divergence, intermediate hidden-state matching, and attention transfer terms",
        "Train the student model on task data using the combined distillation objective with temperature scaling",
        "Apply post-distillation quantization and structured pruning to further reduce memory footprint and latency",
        "Benchmark the compressed student model on-device for accuracy, latency, memory, and energy consumption against the teacher baseline",
    ],
    "unverified_suggestions": ["Edge Impulse", "MLPerf Mobile benchmark suite", "PyTorch Mobile"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly grounds the compression objective in concrete on-device constraints (latency, memory, energy) rather than treating accuracy in isolation.",
        "It combines complementary compression techniques (distillation, quantization, pruning) in a coherent implementation pipeline.",
        "The research questions are well scoped and directly testable against established NLU benchmarks.",
        "It draws on well-established distillation baselines (DistilBERT, TinyBERT, MobileBERT) as points of comparison.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers may be insufficient to fully characterize the state of the art across all distillation loss variants, so conclusions about optimal loss design should be treated as preliminary.",
        "The proposal does not specify which mobile hardware platforms will be used for latency and energy benchmarking, limiting reproducibility.",
        "Dataset fit is uncertain: the listed OpenML repository is generic and not confirmed to include the specific on-device NLP benchmark splits needed for this study.",
        "There is limited discussion of how student architecture search will be constrained computationally, risking an intractable search space.",
    ],
    "missing_components": [
        "A concrete baseline comparison table specifying teacher and student parameter counts across candidate architectures.",
        "An explicit plan for statistical significance testing when comparing distilled model variants.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to compressing NLP models for efficient on-device inference, based on the retrieved abstract." for t in titles]
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
