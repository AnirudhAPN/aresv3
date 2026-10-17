import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Reinforcement Learning for Robotic Manipulation in Unstructured Environments"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep reinforcement learning policies that generalize robotic manipulation skills across cluttered, unstructured environments.",
        "Investigate sample-efficient training strategies that reduce the real-world data requirements for learning manipulation policies.",
        "Evaluate sim-to-real transfer techniques for closing the reality gap in contact-rich manipulation tasks.",
        "Quantify the robustness of learned manipulation policies to novel object geometries, occlusions, and sensor noise.",
    ],
    "research_questions": [
        "How do deep reinforcement learning architectures compare in sample efficiency when learning manipulation policies from high-dimensional visual observations?",
        "What role does domain randomization play in reducing the sim-to-real gap for grasping and manipulation in cluttered scenes?",
        "How can reward shaping and curriculum learning improve convergence for long-horizon, contact-rich manipulation tasks?",
        "To what extent do learned policies generalize to previously unseen object shapes and unstructured clutter configurations?",
    ],
    "keywords": [
        "deep reinforcement learning",
        "robotic manipulation",
        "sim-to-real transfer",
        "grasping in clutter",
        "sample-efficient policy learning",
    ],
    "research_scope": (
        "This research focuses on deep reinforcement learning methods applied to robotic manipulation tasks, "
        "such as grasping, pushing, and object rearrangement, in unstructured and cluttered environments where "
        "object poses, geometries, and scene layouts are not known in advance. The scope includes both "
        "simulation-based training and sim-to-real transfer techniques, covering model-free and model-based "
        "reinforcement learning algorithms operating on visual or proprioceptive observations. Excluded from "
        "this scope are purely classical motion-planning approaches without a learning component, manipulation "
        "in fully structured industrial settings with fixed fixtures, and multi-robot coordination tasks, which "
        "are considered out of scope for this review."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of vision-based end-to-end policies trained via deep reinforcement learning for grasp and manipulation tasks.",
        "Increasing use of domain randomization and procedurally generated simulation environments to improve sim-to-real transfer.",
        "Shift toward sample-efficient off-policy algorithms and offline reinforcement learning to reduce costly real-robot interaction.",
        "Rising interest in combining tactile and proprioceptive feedback with visual inputs for contact-rich manipulation.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarks for comparing sim-to-real transfer performance across heterogeneous manipulation tasks and robot platforms.",
            "based_on_papers": ["Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of policy robustness to long-tail clutter configurations and previously unseen object geometries at scale.",
            "based_on_papers": ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: sample efficiency comparisons that account for the full cost of real-world data collection, including robot downtime and human supervision.",
            "based_on_papers": ["Human-level control through deep reinforcement learning", "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop unified benchmarks and metrics for sim-to-real manipulation transfer across diverse robot embodiments and clutter densities.",
        "Explore hybrid model-based and model-free reinforcement learning approaches to improve sample efficiency for contact-rich tasks.",
        "Integrate multimodal sensing, including tactile feedback, to improve policy robustness under occlusion and unstructured clutter.",
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
            "description": "An off-policy, maximum-entropy actor-critic algorithm well suited to continuous control problems such as robotic manipulation, offering strong sample efficiency and stable training in high-dimensional action spaces.",
            "verified": True,
        },
        {
            "name": "Domain Randomization for Sim-to-Real Transfer",
            "url": "https://arxiv.org/abs/1703.06907",
            "description": "A training technique that randomizes visual and dynamics parameters of the simulator during policy learning so that policies trained purely in simulation transfer more reliably to real robotic manipulators.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Grasp success rate",
        "Sample efficiency (episodes to convergence)",
        "Sim-to-real performance gap",
        "Policy generalization rate on unseen object sets",
    ],
    "implementation_steps": [
        "Define the manipulation task suite, including grasping, pushing, and rearrangement scenarios in cluttered scenes",
        "Build or configure a physics simulator with domain-randomized textures, lighting, and object dynamics",
        "Design the observation and action spaces, incorporating visual, proprioceptive, and where available tactile inputs",
        "Train candidate reinforcement learning policies, such as SAC, in simulation using curriculum learning for long-horizon tasks",
        "Evaluate policies in simulation on held-out clutter configurations and unseen object geometries",
        "Transfer the best-performing policies to a physical robot and measure sim-to-real performance degradation",
    ],
    "unverified_suggestions": [
        "RLBench",
        "Meta-World",
        "Isaac Gym",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly grounds its objectives in well-established reinforcement learning algorithms with verifiable references.",
        "The research scope is precisely bounded, explicitly excluding structured industrial settings and multi-robot coordination.",
        "The methodology includes a coherent, ordered implementation pipeline from simulation setup through real-robot evaluation.",
        "The identified gaps around sim-to-real benchmarking and long-tail clutter robustness are relevant and actionable for future work.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small, so conclusions about broader trends should be treated as preliminary.",
        "The proposal does not fully address dataset fit, since no manipulation-specific benchmark dataset was verified against the whitelist, relying instead on unverified suggestions like RLBench and Meta-World.",
        "Tactile sensing integration is mentioned as a future direction but not incorporated into the concrete implementation steps.",
        "The proposal does not specify compute budget or hardware requirements needed to reproduce the described training pipeline.",
    ],
    "missing_components": [
        "A concrete real-robot hardware platform and sensor specification for the sim-to-real evaluation phase.",
        "An ablation plan isolating the contribution of domain randomization versus algorithmic choice to transfer performance.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep reinforcement learning for robotic manipulation, based on the retrieved abstract." for t in titles]
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
