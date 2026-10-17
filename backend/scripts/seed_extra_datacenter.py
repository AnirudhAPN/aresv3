import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Reinforcement Learning for Data Center Cooling Energy Optimization"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep reinforcement learning (DRL) agent that dynamically controls data center cooling setpoints to minimize energy consumption while maintaining safe thermal operating conditions",
        "Evaluate model-free (SAC, PPO) versus model-based/hybrid RL approaches for sample-efficient cooling control in a simulated data center thermal environment",
        "Investigate safe reinforcement learning techniques, including constrained policy optimization and safety layers, to prevent thermal excursions during online exploration",
        "Assess the transferability of trained cooling control policies across data centers with different physical layouts, sensor placements, and IT load profiles",
    ],
    "research_questions": [
        "Can DRL-based cooling controllers reduce energy consumption relative to traditional PID and rule-based baselines without violating temperature and humidity safety bounds?",
        "Which RL algorithm (SAC, PPO, or a model-based MPC-RL hybrid) achieves the best sample efficiency and stability when trained on a data center thermal simulator?",
        "How effectively do constrained RL formulations and safety layers bound the rate of unsafe control actions during exploration and deployment?",
        "How well do RL cooling policies trained on one facility topology generalize when transferred to an unseen data center layout or sensor configuration?",
    ],
    "keywords": [
        "reinforcement learning data center cooling",
        "HVAC control deep reinforcement learning",
        "energy optimization neural network cooling",
        "safe reinforcement learning constrained control",
        "data center power usage effectiveness",
    ],
    "research_scope": (
        "This research focuses on deep reinforcement learning methods for controlling data center cooling infrastructure, "
        "including chilled water setpoints, fan speeds, and CRAC/CRAH unit control, with the goal of minimizing cooling "
        "energy consumption subject to thermal safety constraints. The scope covers model-free RL algorithms (SAC, PPO), "
        "safe/constrained RL formulations, and simulation-based training using historical telemetry-calibrated thermal "
        "models. It excludes direct physical hardware modification, chip-level thermal design, and cooling technologies "
        "outside conventional air- and liquid-based data center HVAC systems, as well as live production deployment "
        "without a simulation or digital-twin validation stage."
    ),
}

GAP_JSON = {
    "current_trends": [
        "High-profile industrial deployments (e.g., DeepMind's application of RL to Google data center cooling) reporting substantial energy savings, but with limited publicly reproducible benchmarking methodology",
        "Growing use of calibrated digital twin and CFD-informed simulators to pretrain RL agents offline before any live facility deployment",
        "Adoption of safe RL techniques such as constrained Markov decision processes and learned safety layers to bound exploration risk in HVAC control settings",
        "Increasing interest in hybrid model-based control (MPC) combined with RL policy learning to improve sample efficiency over pure model-free approaches",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized, publicly available benchmark thermal simulators for data center cooling RL, comparable to standard control benchmarks, that would allow reproducible cross-paper comparison of algorithms.",
            "based_on_papers": ["Data Center Cooling Using Model-Predictive Control"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: formal safety verification or certification methodology for learned cooling policies prior to deployment in live, safety-critical data center environments.",
            "based_on_papers": ["Data Center Cooling Using Model-Predictive Control"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative evaluation of policy transfer performance when a trained cooling controller is applied to a data center with a different physical layout or sensor topology than the one used for training.",
            "based_on_papers": ["Data Center Cooling Using Model-Predictive Control"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop open, reproducible thermal simulation benchmarks calibrated against real facility telemetry to enable standardized comparison of cooling RL algorithms",
        "Establish formal or empirical safety verification protocols for certifying learned cooling policies before live deployment",
        "Investigate domain-randomization and meta-RL strategies to improve zero-shot policy transfer across differing data center layouts",
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
            "name": "Soft Actor-Critic (SAC)",
            "url": "https://arxiv.org/abs/1801.01290",
            "description": "Off-policy maximum-entropy actor-critic algorithm used as the primary DRL controller for continuous cooling setpoint actions, valued for sample efficiency and training stability.",
            "verified": True,
        },
        {
            "name": "Proximal Policy Optimization (PPO)",
            "url": "https://arxiv.org/abs/1707.06347",
            "description": "On-policy policy-gradient algorithm used as a comparison baseline for the cooling control task, evaluated for training stability and constraint-handling under clipped objective updates.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Power Usage Effectiveness (PUE)",
        "Cumulative cooling energy consumption (kWh)",
        "Thermal constraint violation rate (excursions per episode)",
        "Cumulative episode return / reward",
    ],
    "implementation_steps": [
        "Calibrate a data center thermal simulator using historical facility telemetry (temperature, humidity, IT load, cooling setpoints)",
        "Formulate the cooling control problem as an MDP with state defined by zone temperatures/humidity/IT load and actions defined by chiller and fan setpoints",
        "Implement and tune a baseline rule-based/PID cooling controller for comparison",
        "Train SAC and PPO agents with a constrained optimization or safety-layer mechanism to bound thermal excursions during exploration",
        "Evaluate energy savings and safety violation rates of each trained policy against the PID baseline in simulation",
        "Test zero-shot and fine-tuned transfer of the best-performing policy to a held-out simulated facility layout with a different sensor configuration",
    ],
    "unverified_suggestions": ["ASHRAE data center thermal guideline datasets", "Open Compute Project telemetry logs", "Google data center cooling case study data"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds the RL formulation in a concrete, physically meaningful state-action-reward definition tied to real cooling infrastructure variables",
        "Including a safety-layer or constrained optimization mechanism directly addresses the practical risk of unsafe exploratory actions in a thermally sensitive environment",
        "The evaluation plan compares against both a realistic PID baseline and tests cross-layout transfer, going beyond single-facility benchmarking",
        "Evaluation metrics (PUE, energy consumption, constraint violations) are directly aligned with both operational and safety objectives of the target application",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how the thermal simulator itself will be validated against ground-truth facility behavior, so simulation-to-reality gap remains a significant unaddressed risk",
        "Dataset fit is a concern: the whitelisted repository (OpenML) is unlikely to host facility-specific cooling telemetry, so the unverified suggestions carry most of the practical data-sourcing burden",
        "The plan lacks a discussion of how sensor noise, actuator delay, and non-stationary IT workload patterns (e.g., diurnal or seasonal load shifts) will be modeled, which are known to materially affect real-world cooling control performance",
    ],
    "missing_components": [
        "A concrete plan for validating the calibrated simulator against held-out real facility telemetry before any policy training",
        "An explicit protocol for staged deployment (shadow mode, human-in-the-loop override) prior to full autonomous control in a live facility",
    ],
    "novelty_score": 0.63,
    "quality_score": 0.73,
}

CONFIDENCE_JSON = {"internal_consistency": 0.81, "specificity": 0.79, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to reinforcement learning for data center cooling energy optimization, based on the retrieved abstract." for t in titles]
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
