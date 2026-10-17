import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Reinforcement Learning for Warehouse Robot Path Planning"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep reinforcement learning framework that enables warehouse robots to plan collision-free paths in dynamic, multi-agent environments",
        "Evaluate the trade-offs between sample efficiency and path optimality across different DRL algorithms for warehouse navigation",
        "Investigate how reward shaping and curriculum learning affect convergence speed in cluttered warehouse layouts",
        "Assess the generalization of trained policies to unseen warehouse floor plans and dynamic obstacle configurations",
    ],
    "research_questions": [
        "How do value-based methods (e.g., DQN) compare to policy-gradient methods (e.g., PPO) for multi-robot path planning in warehouses?",
        "What reward function design best balances travel time, energy consumption, and collision avoidance for warehouse robots?",
        "Can curriculum learning reduce the number of training episodes needed for robots to learn efficient paths in dense warehouse layouts?",
        "How well do policies trained in simulation transfer to real-world warehouse robot navigation (sim-to-real gap)?",
    ],
    "keywords": [
        "deep reinforcement learning",
        "warehouse robot navigation",
        "multi-agent path planning",
        "collision avoidance",
        "sim-to-real transfer",
    ],
    "research_scope": "This research focuses on applying deep reinforcement learning (DRL) techniques to the problem of path planning for autonomous mobile robots operating in warehouse environments. The scope includes single-agent and multi-agent formulations, reward function design, and evaluation of sample efficiency, path optimality, and collision avoidance under dynamic obstacle conditions such as moving robots, workers, and shifting inventory. The work also considers the practical constraints of warehouse automation, including partial observability, real-time decision-making requirements, and the challenge of transferring policies learned in simulation to physical robot fleets.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of multi-agent deep reinforcement learning (MARL) for coordinating fleets of warehouse robots rather than treating each robot as an independent planner",
        "Growing use of graph neural networks combined with DRL to encode warehouse topology and inter-robot dependencies for path planning",
        "Shift toward simulation platforms (e.g., NVIDIA Isaac Sim, Gazebo) for pretraining policies before deployment to reduce real-world training cost",
        "Rising interest in hybrid approaches that combine classical planners (A*, RRT) with learned local collision-avoidance policies",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of sim-to-real transfer for DRL-based warehouse path planning under realistic sensor noise and actuator latency",
            "based_on_papers": ["Multi-Agent Deep Reinforcement Learning for Warehouse Robot Coordination"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized benchmarks comparing energy-aware reward shaping across different warehouse layouts and robot fleet sizes",
            "based_on_papers": ["Deep Q-Learning for Autonomous Robot Navigation in Dynamic Environments"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness analysis of learned policies when robot fleet size scales beyond the training distribution",
            "based_on_papers": ["Proximal Policy Optimization for Multi-Robot Task Allocation"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized sim-to-real benchmarks that quantify the performance gap for DRL warehouse navigation policies under sensor and actuation noise",
        "Explore transformer-based attention mechanisms to allow policies to scale to variable numbers of robots without retraining",
        "Integrate energy consumption and battery-aware constraints directly into the reward function for more deployment-realistic policies",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "OpenML",
            "url": "https://www.openml.org/",
            "description": "Searchable repository of ML datasets with standardized benchmarking metadata.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Deep Q-Network (DQN)",
            "url": "https://arxiv.org/abs/1312.5602",
            "description": "A value-based reinforcement learning algorithm that uses a deep neural network to approximate Q-values, well suited as a baseline for discretized grid-based warehouse navigation tasks.",
            "verified": True,
        },
        {
            "name": "Proximal Policy Optimization (PPO)",
            "url": "https://arxiv.org/abs/1707.06347",
            "description": "A policy-gradient algorithm that offers stable, sample-efficient training and is commonly used for continuous-control and multi-agent warehouse robot navigation policies.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Average path length relative to the shortest-path baseline",
        "Collision rate per 1000 navigation episodes",
        "Task completion time (makespan) across the robot fleet",
        "Policy convergence speed measured in training episodes to reach a target success rate",
    ],
    "implementation_steps": [
        "Construct a simulated warehouse environment with configurable grid layouts, shelving units, and dynamic obstacles using a platform such as Gazebo or Isaac Sim",
        "Define the state representation (e.g., local occupancy grid, robot pose, goal position) and action space (discrete moves or continuous velocity commands) for each robot",
        "Implement baseline DQN and PPO agents with a shared reward function penalizing collisions and rewarding progress toward the goal",
        "Train agents using curriculum learning, progressively increasing obstacle density and fleet size across training stages",
        "Evaluate trained policies on held-out warehouse layouts not seen during training to assess generalization",
        "Deploy the best-performing policy on a physical or high-fidelity simulated robot to measure the sim-to-real performance gap",
    ],
    "unverified_suggestions": [
        "Consider incorporating graph neural network layers to encode inter-robot spatial relationships for improved multi-agent coordination",
        "Explore reward decomposition techniques to separately optimize for collision avoidance and travel efficiency",
        "Investigate domain randomization during simulation training to improve robustness to real-world sensor noise",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The research questions clearly separate single-agent and multi-agent formulations, enabling incremental complexity in experimentation",
        "The methodology proposes both value-based and policy-gradient baselines, allowing for a meaningful algorithmic comparison",
        "The evaluation metrics capture both efficiency (path length, makespan) and safety (collision rate), which are the two dominant concerns in warehouse robotics",
        "The inclusion of curriculum learning and sim-to-real evaluation reflects awareness of practical deployment challenges beyond pure simulation performance",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small and may not sufficiently cover recent graph-neural-network-based approaches to multi-robot planning",
        "The proposal does not specify how many robots will be simulated concurrently, which materially affects the difficulty of the multi-agent coordination problem",
        "There is no discussion of computational budget or hardware requirements for training, which are significant for DRL experiments at this scale",
        "The dataset section relies on a generic ML repository (OpenML) rather than a warehouse-robotics-specific simulation benchmark, suggesting a potential mismatch between proposed datasets and the stated research questions",
    ],
    "missing_components": [
        "A concrete baseline comparison against classical (non-learning) path planners such as A* or Dijkstra to contextualize DRL gains",
        "An ablation study isolating the contribution of curriculum learning versus reward shaping to overall policy performance",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep reinforcement learning for warehouse robot path planning, based on the retrieved abstract." for t in titles]
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
