import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Circuit Design Optimization in Electronic Design Automation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop graph neural network models that represent digital and analog circuit netlists as graphs to predict and optimize performance metrics such as timing, power, and area",
        "Evaluate GNN-based surrogate models for accelerating placement and routing optimization compared to traditional EDA tool heuristics",
        "Investigate reinforcement learning combined with GNN circuit representations for automated chip floorplanning and macro placement",
        "Assess the generalization of GNN-based circuit optimization models across different circuit topologies and semiconductor technology nodes",
    ],
    "research_questions": [
        "Can GNN-based surrogate models accurately predict post-layout timing, power, and area metrics directly from pre-layout netlist graphs, reducing reliance on costly full EDA simulation cycles?",
        "How does combining GNN circuit representations with reinforcement learning for macro placement compare to traditional simulated-annealing-based placement methods?",
        "To what extent do GNN circuit optimization models trained on one technology node or circuit family generalize to unseen topologies and process nodes?",
        "Which GNN architectural choices (message-passing depth, homogeneous versus heterogeneous graph representations distinguishing gates, nets, and modules) most affect prediction accuracy for circuit performance metrics?",
    ],
    "keywords": [
        "graph neural network circuit design",
        "chip placement reinforcement learning",
        "electronic design automation machine learning",
        "netlist graph representation learning",
        "GNN timing power area prediction",
    ],
    "research_scope": (
        "This research focuses on applying graph neural networks to represent circuit netlists for predicting and "
        "optimizing timing, power, and area (PPA) metrics in electronic design automation, covering GNN surrogate "
        "modeling for pre-layout PPA estimation and reinforcement-learning-driven macro placement using learned "
        "graph embeddings. It includes evaluation of generalization across circuit topologies and technology nodes "
        "using open-source benchmark netlists. The scope excludes physical fabrication process modeling, analog "
        "circuit sizing at the transistor level, and full commercial EDA tool replacement, which are treated as "
        "future engineering directions beyond this work."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of GNNs to encode circuit netlists as graphs for predicting timing, power, and area metrics without requiring full physical synthesis",
        "Application of reinforcement learning combined with graph representations for automated macro placement and floorplanning, following approaches such as Google's learned chip placement methodology",
        "Adoption of heterogeneous graph representations that distinguish gate types, nets, and hierarchical modules to improve circuit GNN expressiveness",
        "Interest in using GNN surrogate models to accelerate design space exploration in place of expensive commercial EDA tool runs",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: independent reproducibility and generalization evaluation of RL-based chip placement methods across diverse open-source circuit benchmarks beyond the originally reported proprietary designs.",
            "based_on_papers": ["A graph placement methodology for fast chip design"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic comparison of homogeneous versus heterogeneous GNN circuit representations for PPA prediction accuracy across multiple technology nodes.",
            "based_on_papers": ["Graph Attention Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: transferability evaluation of GNN-based PPA surrogate models when applied to circuit topologies or technology nodes not seen during training.",
            "based_on_papers": ["A graph placement methodology for fast chip design"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Establish open, reproducible benchmarks for RL-based chip placement across diverse public circuit designs to enable independent verification of reported results",
        "Systematically compare homogeneous and heterogeneous graph representations for PPA prediction across multiple open technology node PDKs",
        "Develop domain-adaptation or meta-learning techniques for transferring GNN PPA surrogate models across unseen circuit topologies and technology nodes",
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
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Graph neural network using learned attention coefficients over neighboring gates and nets, used to embed circuit netlist graphs for downstream PPA prediction.",
            "verified": True,
        },
        {
            "name": "GraphSAGE",
            "url": "https://arxiv.org/abs/1706.02216",
            "description": "Inductive graph representation learning framework that aggregates sampled neighborhood features, used to generalize netlist embeddings to previously unseen circuit topologies during evaluation.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Timing prediction error (correlation and MAE versus signoff static timing analysis)",
        "Power prediction error (MAE)",
        "Post-placement wirelength/area metric",
        "Placement runtime speedup versus traditional EDA flow",
    ],
    "implementation_steps": [
        "Parse open-source circuit netlists into graph representations with gate and net nodes and connectivity edges",
        "Extract node and edge features (gate type, fan-in/fan-out, estimated wire length, cell area)",
        "Train a GNN surrogate model to predict post-layout timing, power, and area from the pre-layout netlist graph",
        "Integrate learned GNN embeddings into a reinforcement-learning-based placement agent for macro and floorplan optimization",
        "Benchmark placement quality and runtime against simulated-annealing and open-source EDA placement baselines",
        "Evaluate cross-technology-node and cross-topology generalization on a held-out set of circuit benchmarks",
    ],
    "unverified_suggestions": ["OpenROAD/OpenCores benchmark netlists", "ISPD placement contest benchmarks", "EPFL combinational benchmark suite"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds GNN circuit representation learning in a concrete, practically valuable objective: reducing reliance on costly full-synthesis EDA simulation cycles",
        "Combining GNN embeddings with reinforcement learning for placement directly builds on and critically examines a well-known but only partially reproducible prior approach",
        "The generalization evaluation across technology nodes and topologies addresses a genuine, underexplored limitation of prior GNN-EDA work",
        "Using open-source netlist benchmarks and EDA tools (rather than proprietary designs) supports reproducibility, directly responding to the identified gap",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how signoff-quality ground truth (timing, power) will be obtained for training labels, which typically requires access to commercial EDA tools with restrictive licensing",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host circuit netlists, so feasibility depends entirely on the unverified open-source EDA benchmark suggestions and their licensing/tooling compatibility",
        "The plan does not address how differences in physical design rules and standard cell libraries across technology nodes will be normalized before claiming cross-node transferability, which is a substantial methodological complication specific to this domain",
    ],
    "missing_components": [
        "A concrete plan for sourcing signoff-quality timing/power labels without dependence on restrictive commercial EDA tool licenses",
        "A normalization strategy for standard cell library and design rule differences when evaluating cross-technology-node generalization",
    ],
    "novelty_score": 0.75,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.8, "specificity": 0.78, "completeness": 0.84}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph neural networks for circuit design optimization in electronic design automation, based on the retrieved abstract." for t in titles]
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
