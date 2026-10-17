import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Architecture Search for Energy-Efficient Edge AI Deployment"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a neural architecture search (NAS) framework that jointly optimizes predictive accuracy and on-device energy consumption for edge AI deployment.",
        "Quantify the trade-offs between model latency, memory footprint, and energy usage across representative edge hardware platforms (microcontrollers, mobile SoCs, TPU/NPU accelerators).",
        "Design hardware-aware search spaces and proxy metrics that reduce the computational cost of NAS while preserving correlation with real deployed-device measurements.",
        "Evaluate the generalization of energy-efficient architectures found via NAS across diverse edge tasks such as image classification, keyword spotting, and sensor-based activity recognition.",
    ],
    "research_questions": [
        "How can energy consumption be reliably estimated or measured during the NAS search loop without prohibitive overhead?",
        "What search space design choices (e.g., depthwise separable convolutions, dynamic channel pruning) most directly influence the energy-accuracy Pareto frontier on edge devices?",
        "To what extent do architectures discovered via hardware-aware NAS transfer across different edge accelerators without re-searching?",
        "What are the practical limits of NAS-derived models compared to hand-designed efficient architectures (e.g., MobileNet, EfficientNet-Lite) under strict energy budgets?",
    ],
    "keywords": [
        "neural architecture search",
        "edge AI energy efficiency",
        "hardware-aware NAS",
        "TinyML model compression",
        "on-device inference latency",
    ],
    "research_scope": (
        "This research focuses on neural architecture search techniques specifically targeting energy-constrained edge "
        "inference scenarios, including microcontrollers, smartphones, and embedded accelerators. It covers search "
        "strategies (reinforcement learning, evolutionary, differentiable/gradient-based, and one-shot supernet approaches), "
        "hardware-aware proxies for energy and latency, and evaluation on standard edge benchmarks such as image "
        "classification and keyword spotting. It excludes NAS work aimed purely at cloud/datacenter-scale accuracy "
        "maximization without energy or latency constraints, and does not cover architecture search for non-neural "
        "models or training-time (rather than inference-time) energy optimization."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of differentiable NAS (e.g., DARTS-style) and one-shot supernet training to amortize search cost across many candidate edge architectures.",
        "Increasing use of hardware-in-the-loop or learned latency/energy predictors as auxiliary objectives alongside accuracy during search.",
        "Convergence toward mobile-friendly building blocks (depthwise separable convolutions, squeeze-and-excitation, inverted residuals) as the dominant NAS search space primitives for edge targets.",
        "Emerging interest in multi-objective and once-for-all search methods that produce a single supernet from which many device-specific sub-networks can be extracted without re-training.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized, reproducible on-device energy measurement protocols that would allow direct comparison of NAS methods across different edge hardware platforms.",
            "based_on_papers": ["MnasNet: Platform-Aware Neural Architecture Search for Mobile"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of NAS-discovered architectures to quantization and pruning applied after the search, particularly for ultra-low-power microcontroller deployment.",
            "based_on_papers": ["MCUNet: Tiny Deep Learning on IoT Devices"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: transferability studies quantifying how much accuracy or energy efficiency is lost when a NAS-found architecture from one edge accelerator is redeployed on a different accelerator without re-searching.",
            "based_on_papers": ["Once-for-All: Train One Network and Specialize it for Efficient Deployment"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Development of open, cross-platform energy measurement benchmarks to enable fair comparison of hardware-aware NAS methods.",
        "Joint search over architecture, quantization bit-width, and pruning schedule as a unified multi-objective optimization problem.",
        "Meta-learning or transfer-learning approaches that allow a single searched supernet to specialize efficiently to new, previously unseen edge hardware targets.",
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
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://arxiv.org/abs/1409.1556",
            "description": "Baseline deep architecture family used as the backbone search space for image classification tasks targeted by hardware-aware NAS.",
            "verified": True,
        },
        {
            "name": "Differentiable Architecture Search (DARTS)",
            "url": "https://arxiv.org/abs/1806.09055",
            "description": "Gradient-based NAS method that relaxes the discrete search space to be continuous, enabling efficient joint optimization of architecture and energy/latency objectives.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Top-1 classification accuracy",
        "Energy consumption per inference (mJ)",
        "Inference latency (ms) on target edge hardware",
        "Peak memory footprint (KB/MB)",
    ],
    "implementation_steps": [
        "Define a hardware-aware search space of mobile-friendly building blocks (depthwise separable convolutions, inverted residuals, squeeze-and-excitation modules).",
        "Instrument target edge devices with power measurement tooling to collect ground-truth energy and latency data for a sample of candidate architectures.",
        "Train a lightweight energy/latency predictor model on the collected hardware measurements to serve as a fast proxy during search.",
        "Run a multi-objective search (evolutionary or differentiable) that jointly optimizes validation accuracy and the predicted energy/latency cost.",
        "Fine-tune and quantize the top Pareto-optimal candidate architectures, then re-validate energy and accuracy on physical edge hardware.",
        "Benchmark the final selected architecture against established efficient baselines such as MobileNetV3 and EfficientNet-Lite under identical energy budgets.",
    ],
    "unverified_suggestions": ["Google Edge TPU Benchmark Suite", "MLPerf Tiny", "NVIDIA Jetson Power Estimator"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly articulates a multi-objective optimization framing that balances accuracy against real energy constraints rather than treating efficiency as an afterthought.",
        "The implementation plan is concrete and sequenced, moving logically from search-space definition to hardware-in-the-loop validation.",
        "Grounding the search space in well-established mobile-friendly primitives (depthwise separable convolutions, inverted residuals) increases the likelihood of practical, deployable results.",
        "The inclusion of a learned energy/latency predictor addresses a genuine bottleneck in hardware-aware NAS, namely the cost of repeated on-device measurement during search.",
    ],
    "weaknesses": [
        "The evidence base is limited to a small set of retrieved papers, so claims about gaps such as cross-platform transferability should be treated as preliminary rather than conclusive.",
        "OpenML, the only verified dataset source, is not a strong fit for edge-specific benchmarks (e.g., keyword spotting, IoT sensor streams), so dataset selection needs further validation against domain-specific corpora.",
        "The proposal does not fully specify how variance in power measurement across device units and thermal conditions will be controlled for during evaluation.",
        "There is limited discussion of the computational cost of the search process itself, which could offset the energy savings gained by the resulting deployed model.",
    ],
    "missing_components": [
        "A concrete baseline comparison protocol specifying which hand-designed architectures and NAS competitors will be benchmarked under matched energy budgets.",
        "An ablation plan isolating the contribution of the learned energy predictor versus direct hardware-in-the-loop measurement.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to energy-efficient edge AI architecture search, based on the retrieved abstract." for t in titles]
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
