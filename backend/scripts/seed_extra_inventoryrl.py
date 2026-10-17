import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Reinforcement Learning for Inventory Management in E-Commerce Warehouses"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep reinforcement learning agents that learn replenishment and stocking policies for e-commerce warehouse inventory under stochastic demand and lead-time uncertainty",
        "Evaluate policy-gradient and value-based deep RL algorithms against classical inventory control baselines (e.g., (s,S) policies, base-stock policies) for cost and service-level performance",
        "Investigate multi-echelon inventory coordination, where RL agents must jointly manage stock across a warehouse network rather than a single-node setting",
        "Assess the robustness of learned inventory policies to demand distribution shift, such as seasonal spikes and promotional events not seen during training",
    ],
    "research_questions": [
        "Do deep RL inventory policies achieve lower total cost (holding, stockout, ordering) than classical (s,S) or base-stock policies under matched demand scenarios?",
        "How do policy-gradient methods (e.g., PPO) compare to value-based methods (e.g., DQN) for inventory replenishment in terms of sample efficiency and policy stability?",
        "Can a single RL policy coordinate replenishment decisions across a multi-echelon warehouse network more effectively than independently optimized per-node policies?",
        "How robust are RL-learned inventory policies to demand distribution shifts, such as unseen seasonal or promotional demand spikes, relative to classical policies?",
    ],
    "keywords": [
        "deep reinforcement learning inventory management",
        "warehouse replenishment optimization",
        "supply chain reinforcement learning",
        "multi-echelon inventory control",
        "e-commerce demand forecasting",
    ],
    "research_scope": (
        "This research covers deep reinforcement learning methods for inventory replenishment and stocking decisions "
        "in e-commerce warehouse settings, including single-node and multi-echelon coordination under stochastic "
        "demand and lead-time uncertainty, benchmarked against classical inventory control policies. It excludes "
        "warehouse robotics and physical path planning, last-mile delivery routing, and pricing or demand-generation "
        "optimization that is not directly tied to inventory replenishment decisions."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of policy-gradient methods such as PPO for inventory replenishment tasks, favored for their stability over earlier value-based approaches in continuous or high-dimensional action settings",
        "Increasing interest in multi-echelon and multi-agent RL formulations that coordinate replenishment decisions across warehouse networks rather than single-node settings",
        "Adoption of simulation environments (e.g., OR-Gym) that standardize inventory management benchmarks for reproducible RL evaluation",
        "Emerging focus on robustness testing under demand distribution shift, reflecting real e-commerce seasonality and promotional demand spikes",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a head-to-head comparison of policy-gradient and value-based deep RL methods on the same multi-echelon inventory benchmark under matched cost structures.",
            "based_on_papers": ["A Deep Reinforcement Learning Approach for Inventory Management"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of RL policy robustness to out-of-distribution demand spikes (seasonal or promotional) not represented during training.",
            "based_on_papers": ["Solving the Supply Chain Optimization Problem Using Deep Reinforcement Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: analysis of how learned multi-echelon coordination policies compare to independently optimized per-node classical policies under identical network topologies.",
            "based_on_papers": ["A Deep Reinforcement Learning Approach for Inventory Management"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish standardized multi-echelon inventory benchmarks that support direct comparison of policy-gradient and value-based RL methods",
        "Develop demand-shift stress-testing protocols that evaluate RL inventory policies on held-out seasonal and promotional demand patterns",
        "Study hybrid approaches that combine classical inventory theory guarantees with learned RL policies for improved worst-case robustness",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Proximal Policy Optimization (PPO)",
            "url": "https://arxiv.org/abs/1707.06347",
            "description": "Policy-gradient reinforcement learning algorithm used to learn stable replenishment policies under stochastic demand and lead-time uncertainty.",
            "verified": True,
        },
        {
            "name": "Deep Q-Network (DQN)",
            "url": "https://arxiv.org/abs/1312.5602",
            "description": "Value-based deep reinforcement learning algorithm used as a comparison baseline for discretized inventory ordering decisions.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Total inventory cost (holding + stockout + ordering cost)",
        "Service level (fill rate) achieved per SKU",
        "Policy performance gap relative to classical (s,S) or base-stock baselines",
        "Cost degradation under out-of-distribution demand spike scenarios",
    ],
    "implementation_steps": [
        "Build a simulation environment modeling warehouse inventory dynamics, including stochastic demand, lead times, holding costs, and stockout penalties",
        "Extend the simulation to a multi-echelon network topology representing distribution centers and downstream fulfillment nodes",
        "Implement PPO and DQN agents to learn replenishment policies, alongside classical (s,S) and base-stock policy baselines tuned on the same cost structure",
        "Train agents across a range of demand distributions (e.g., Poisson, seasonal, promotional-spike) representative of e-commerce demand patterns",
        "Evaluate learned policies on held-out demand scenarios, including out-of-distribution seasonal and promotional spikes not seen during training",
        "Compare multi-echelon coordinated RL policies against independently optimized per-node classical policies on identical network topologies",
    ],
    "unverified_suggestions": ["OR-Gym inventory management simulation benchmark", "RetailRocket e-commerce dataset", "Instacart Market Basket dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly benchmarks deep RL against classical inventory control theory rather than only against other RL variants, grounding novelty claims in a well-established baseline",
        "Including multi-echelon coordination as an explicit objective addresses a more operationally realistic setting than the common single-node inventory formulation",
        "The demand distribution-shift robustness evaluation targets a practically important failure mode (seasonality, promotions) often ignored in RL inventory studies",
        "Comparing policy-gradient and value-based methods head-to-head on a shared benchmark provides a clear, reproducible experimental contribution",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "Dataset fit is uncertain: the whitelisted repository (OpenML) does not host ready-made multi-echelon inventory simulation environments, so OR-Gym and related suggestions remain unverified and would need to be integrated separately.",
        "The proposal relies on simulated demand and lead-time dynamics, and it does not address how well simulation-trained policies would transfer to real warehouse operations with unmodeled supplier variability or SKU-level correlations.",
        "There is limited discussion of the computational cost and training stability of RL policies at the scale of a realistic SKU catalog (thousands to millions of items), which is a significant practical concern for e-commerce deployment.",
    ],
    "missing_components": [
        "A sim-to-real validation step or discussion of transferring simulation-trained policies to pilot warehouse operations.",
        "A scalability analysis addressing training and inference cost at realistic e-commerce SKU-catalog sizes.",
    ],
    "novelty_score": 0.70,
    "quality_score": 0.78,
}

CONFIDENCE_JSON = {"internal_consistency": 0.80, "specificity": 0.82, "completeness": 0.84}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep reinforcement learning for e-commerce warehouse inventory management, based on the retrieved abstract." for t in titles]
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
