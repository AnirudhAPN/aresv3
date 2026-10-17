import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer Models for Multi-Hop Question Answering over Knowledge Graphs"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a transformer-based model that answers multi-hop questions by reasoning over paths in a knowledge graph combined with textual context",
        "Evaluate graph-aware attention mechanisms, including relation-aware self-attention and GNN-transformer hybrids, for their ability to traverse multi-hop reasoning chains accurately",
        "Investigate how explicit path retrieval and ranking versus end-to-end implicit reasoning affects multi-hop QA accuracy and answer explainability",
        "Assess model robustness to incomplete or noisy knowledge graphs, a common condition in real-world knowledge graph deployments",
    ],
    "research_questions": [
        "Do hybrid GNN-transformer architectures that explicitly encode knowledge graph structure outperform text-only transformer QA models on multi-hop reasoning benchmarks?",
        "How does explicit multi-hop path retrieval and ranking affect both answer accuracy and the interpretability of the reasoning chain compared to end-to-end implicit reasoning?",
        "How robust are transformer-based multi-hop QA models to missing edges or noisy relations in the underlying knowledge graph?",
        "What is the tradeoff between reasoning-chain length, measured in number of hops, and answer accuracy, and where do current models' failure rates increase sharply?",
    ],
    "keywords": [
        "multi-hop question answering knowledge graph",
        "graph-aware transformer reasoning",
        "knowledge graph embedding QA",
        "relation-aware attention",
        "complex question answering over KG",
    ],
    "research_scope": (
        "This research covers transformer-based and hybrid GNN-transformer architectures for multi-hop "
        "question answering that requires traversing multiple relations and entities in a knowledge graph, "
        "including explicit path retrieval and end-to-end implicit reasoning approaches. The scope excludes "
        "single-hop factoid QA, purely text-based multi-hop QA over unstructured document sets without a "
        "knowledge graph, and knowledge graph construction or completion as a primary objective."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Hybrid architectures combining graph neural network encoders with transformer reasoning layers, such as relation-aware self-attention over retrieved knowledge graph subgraphs",
        "Growing use of explicit path retrieval and re-ranking modules that identify candidate reasoning paths before answer generation",
        "Adoption of large pretrained language models fine-tuned or prompted with linearized knowledge graph triples for multi-hop reasoning",
        "Increasing focus on robustness and faithfulness evaluation, checking whether generated reasoning chains reflect genuine knowledge graph traversal rather than shortcut pattern matching",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of explicit path-retrieval-then-reason pipelines versus end-to-end implicit GNN-transformer reasoning under matched multi-hop benchmarks.",
            "based_on_papers": ["HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", "Knowledge Graph Embedding for Link Prediction: A Comparative Analysis"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness evaluation under systematically injected knowledge graph incompleteness or noise, since most benchmarks assume a fixed, relatively clean knowledge graph.",
            "based_on_papers": ["HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: verification of whether generated reasoning chains are faithful to actual multi-hop knowledge graph traversal rather than exploiting shortcut correlations.",
            "based_on_papers": ["Are Multi-Hop Reading Comprehension Models Learning Multi-hop Reasoning?"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop knowledge-graph-incompleteness robustness benchmarks that systematically vary the fraction of missing or noisy edges",
        "Design faithfulness evaluation protocols that verify generated reasoning chains against ground-truth knowledge graph paths rather than final-answer accuracy alone",
        "Explore unified architectures that jointly optimize path retrieval and answer generation rather than treating them as separate pipeline stages",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Attention-based graph neural network used as the base for relation-aware extensions that encode knowledge graph subgraph structure feeding into the transformer reasoning layer.",
            "verified": True,
        },
        {
            "name": "BERT",
            "url": "https://arxiv.org/abs/1810.04805",
            "description": "Pretrained transformer encoder used as the base reading-comprehension and reasoning module that combines retrieved textual context with knowledge graph path information.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Exact Match (EM)",
        "F1-score on predicted answer span",
        "Supporting-fact / reasoning-path retrieval F1",
        "Reasoning-chain faithfulness accuracy",
    ],
    "implementation_steps": [
        "Assemble a multi-hop QA benchmark with paired knowledge graph triples and, where available, supporting textual passages and ground-truth reasoning paths",
        "Implement an explicit path-retrieval module that ranks candidate multi-hop reasoning paths in the knowledge graph relevant to the question",
        "Implement a hybrid GNN-transformer architecture that encodes retrieved subgraphs with relation-aware attention and generates the final answer",
        "Implement an end-to-end implicit reasoning baseline that omits explicit path retrieval for comparison",
        "Evaluate all variants on EM, F1, and supporting-fact retrieval metrics across questions stratified by required hop count",
        "Systematically inject missing or noisy edges into the knowledge graph and re-evaluate all models to assess robustness degradation",
    ],
    "unverified_suggestions": ["HotpotQA", "MetaQA", "WebQuestionsSP"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal poses a well-defined architectural comparison, explicit path retrieval versus implicit end-to-end reasoning, with concrete, measurable evaluation criteria",
        "Stratifying evaluation by required hop count directly targets a known and important failure mode of multi-hop QA systems",
        "The inclusion of a reasoning-chain faithfulness metric goes beyond final-answer accuracy, addressing a recognized weakness in prior multi-hop QA evaluation practice",
        "Robustness testing under systematically injected knowledge graph incompleteness reflects realistic deployment conditions rather than assuming a clean, complete KG",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how ground-truth reasoning paths will be obtained or validated for datasets that provide only supporting facts rather than explicit knowledge graph paths",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host multi-hop QA or knowledge graph benchmarks such as HotpotQA or MetaQA, so those remain unverified suggestions requiring separate access",
        "The plan does not address computational cost and scalability of subgraph retrieval over large real-world knowledge graphs, which can be a significant bottleneck distinct from the smaller benchmark KGs used in evaluation",
    ],
    "missing_components": [
        "A defined procedure for deriving or validating ground-truth reasoning paths from datasets that only provide supporting facts rather than explicit KG traversals",
        "A scalability analysis of subgraph retrieval cost on large real-world knowledge graphs versus the smaller benchmark KGs used for evaluation",
    ],
    "novelty_score": 0.74,
    "quality_score": 0.79,
}

CONFIDENCE_JSON = {"internal_consistency": 0.86, "specificity": 0.83, "completeness": 0.88}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based multi-hop question answering over knowledge graphs, based on the retrieved abstract." for t in titles]
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
