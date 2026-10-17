import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Federated Learning for Next-Word Prediction on Mobile Keyboards"

PLANNER_JSON = {
    "research_objectives": [
        "Design a federated learning system that trains next-word prediction models directly on mobile devices without centralizing raw keystroke data",
        "Evaluate the trade-off between prediction accuracy and communication/computation cost under realistic mobile hardware constraints",
        "Assess the effect of non-IID user typing data and client heterogeneity on convergence of the global language model",
        "Integrate differential privacy and secure aggregation to quantify the privacy-utility trade-off for on-device keyboard personalization",
    ],
    "research_questions": [
        "How does Federated Averaging compare to centralized training in next-word prediction accuracy on mobile keyboard datasets?",
        "What is the impact of non-IID client data distributions on the convergence speed and final perplexity of the federated language model?",
        "How much does adding differential privacy noise degrade next-word prediction accuracy, and at what privacy budget does this become unacceptable?",
        "Can model compression or quantization reduce the communication cost of federated rounds without significant loss in prediction quality?",
    ],
    "keywords": [
        "federated learning",
        "next-word prediction",
        "mobile keyboard personalization",
        "on-device NLP",
        "differential privacy",
    ],
    "research_scope": "This research focuses on applying federated learning to the task of next-word prediction for mobile keyboard applications, where user keystroke data remains on-device and only model updates are communicated to a central server. The scope covers algorithmic approaches such as Federated Averaging and its variants, lightweight recurrent and transformer-based language models suitable for mobile inference, and privacy-preserving mechanisms including differential privacy and secure aggregation. The study also considers practical deployment constraints such as limited battery, bandwidth, and compute on edge devices, and evaluates methods against both prediction quality metrics and system-level efficiency metrics, drawing on public benchmarks such as the Reddit and Stack Overflow federated text datasets as proxies for real keyboard usage.",
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of Federated Averaging (FedAvg) and its communication-efficient variants (FedProx, FedOpt) as the default optimization framework for on-device language modeling",
        "Increasing use of lightweight LSTM and small transformer architectures to fit next-word prediction models within mobile memory and latency budgets",
        "Growing integration of differential privacy and secure aggregation protocols to provide formal privacy guarantees for keystroke-derived training data",
        "Shift toward personalization layers on top of a shared global model to account for individual user vocabulary and typing style",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of federated next-word prediction under realistic device dropout and intermittent connectivity patterns seen in real mobile deployments.",
            "based_on_papers": ["Federated Learning for Mobile Keyboard Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative comparison of the accuracy-privacy trade-off across a range of differential privacy budgets specifically for next-word prediction tasks rather than image classification benchmarks.",
            "based_on_papers": ["Learning Differentially Private Recurrent Language Models"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: energy and battery-consumption measurements for on-device federated training rounds on actual low-end and mid-range Android hardware.",
            "based_on_papers": ["Federated Learning: Strategies for Improving Communication Efficiency"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop adaptive client sampling strategies that account for battery level, network type, and typing activity to reduce dropout-induced bias in federated rounds",
        "Explore hybrid personalization approaches that combine a shared federated base model with lightweight on-device fine-tuning to better capture individual vocabulary",
        "Benchmark energy and latency costs of federated training directly on consumer mobile hardware to complement existing simulation-based evaluations",
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
        },
        {
            "name": "LEAF Federated Learning Benchmark",
            "url": "https://leaf.cmu.edu/",
            "description": "A benchmarking framework providing federated datasets including a Reddit next-word prediction corpus partitioned by user, widely used as a proxy for mobile keyboard data.",
            "verified": True,
        },
    ],
    "algorithms": [
        {
            "name": "Federated Averaging (FedAvg)",
            "url": "https://arxiv.org/abs/1602.05629",
            "description": "The baseline federated optimization algorithm that averages locally trained client model updates on a central server, forming the standard training procedure for on-device next-word prediction models.",
            "verified": True,
        },
        {
            "name": "Coupled Input-Forget Gate LSTM Language Model",
            "url": "https://arxiv.org/abs/1611.09805",
            "description": "A compact recurrent language model architecture with reduced parameter count that is well suited for on-device inference and federated training under mobile memory constraints.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Top-1 and top-3 next-word prediction accuracy on held-out client data",
        "Perplexity of the federated language model relative to a centrally trained baseline",
        "Number of communication rounds required to reach target accuracy",
        "Per-round communication cost in megabytes uploaded and downloaded per client",
    ],
    "implementation_steps": [
        "Partition a public text corpus (e.g., the LEAF Reddit dataset) by simulated user to emulate non-IID mobile keyboard usage patterns",
        "Implement a lightweight LSTM or small transformer next-word prediction model sized for on-device inference",
        "Implement the FedAvg training loop with configurable client sampling rate, local epochs, and learning rate schedule",
        "Add differential privacy via per-client gradient clipping and Gaussian noise injection before server aggregation",
        "Train the model across simulated federated rounds and log accuracy, perplexity, and communication cost at each round",
        "Compare federated results against a centrally trained model on the same data split to quantify the utility gap",
    ],
    "unverified_suggestions": [
        "Evaluate secure aggregation protocols to prevent the server from inspecting individual client updates before averaging",
        "Prototype the pipeline on an actual Android test device to measure real battery and thermal impact of local training",
        "Explore knowledge distillation from a larger centrally pretrained language model into the on-device federated model to bootstrap early-round accuracy",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research questions that connect federated optimization theory to a concrete, widely used mobile application",
        "Methodology grounds evaluation in an established federated benchmark (LEAF) rather than an ad hoc data split, improving reproducibility",
        "Explicit inclusion of differential privacy alongside accuracy metrics reflects the real privacy-utility trade-off practitioners face",
        "Implementation steps are concrete and sequenced in a way that could be directly translated into a working experimental pipeline",
    ],
    "weaknesses": [
        "The proposal relies on the LEAF Reddit corpus as a proxy for mobile keyboard input, but this dataset does not capture true on-device typing patterns, abbreviations, or autocorrect interactions, so evidence for real-world applicability remains limited",
        "No baseline comparison against non-federated on-device personalization (e.g., simple per-user n-gram caching) is included, making it harder to isolate the specific benefit of federation",
        "Communication cost analysis does not account for model compression techniques that are commonly paired with FedAvg in production keyboard systems",
        "The privacy evaluation considers only differential privacy and omits discussion of secure aggregation's effect on the same accuracy metrics",
    ],
    "missing_components": [
        "A discussion of client dropout and partial participation effects on convergence guarantees",
        "A power/latency measurement plan for validating feasibility on actual mobile hardware rather than simulation alone",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.81, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to federated on-device next-word prediction, based on the retrieved abstract." for t in titles]
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
