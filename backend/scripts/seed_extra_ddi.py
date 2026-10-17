import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Neural Networks for Drug-Drug Interaction Prediction"

PLANNER_JSON = {
    "research_objectives": [
        "Develop graph neural network models that predict drug-drug interactions by representing drugs as nodes in a molecular or interaction graph and interactions as edges to be predicted",
        "Evaluate graph convolutional and graph attention architectures against traditional similarity-based and matrix factorization baselines for interaction link prediction",
        "Investigate how incorporating molecular substructure information alongside interaction network topology improves prediction of previously unseen drug pairs",
        "Assess model performance specifically on cold-start drug pairs, where at least one drug has few or no known interactions in the training graph",
    ],
    "research_questions": [
        "Do graph neural network architectures outperform similarity-based and matrix factorization methods for predicting unobserved drug-drug interactions?",
        "How does combining molecular substructure features with interaction network topology affect prediction accuracy compared to using either signal alone?",
        "How well do graph neural network models generalize to cold-start drugs with few or no known interactions in the training data?",
        "What is the effect of multi-relational edge types (e.g., specific interaction mechanisms such as pharmacokinetic versus pharmacodynamic effects) on prediction performance compared to a single binary interaction edge?",
    ],
    "keywords": [
        "graph neural network drug-drug interaction",
        "graph convolutional network link prediction",
        "polypharmacy side effect prediction",
        "molecular graph representation learning",
        "drug interaction knowledge graph",
    ],
    "research_scope": (
        "This research focuses on graph neural network methods for predicting drug-drug interactions, representing "
        "drugs as nodes in molecular structure graphs and/or interaction networks and framing interaction prediction "
        "as a link or edge-type prediction task. The scope covers graph convolutional and graph attention "
        "architectures, fusion of molecular substructure features with network topology, multi-relational edge "
        "modeling, and cold-start generalization to drugs with sparse known interactions. It excludes de novo drug "
        "design and molecule generation, wet-lab pharmacokinetic experimentation, and protein structure prediction, "
        "which are treated as out of scope."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of graph convolutional networks over multi-relational drug interaction graphs, following influential work such as Decagon on modeling polypharmacy side effects",
        "Increasing incorporation of molecular substructure graphs alongside interaction network topology to improve generalization to novel drug pairs",
        "Rising adoption of graph attention mechanisms to weight the relative importance of neighboring drugs or substructures during message passing",
        "Continued reliance on a small set of public benchmark graphs (e.g., DrugBank-derived and TWOSIDES interaction networks) for evaluation across studies",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of graph neural network drug-drug interaction models specifically on cold-start drug pairs where one or both drugs have very few known training interactions, as opposed to random edge-holdout splits that favor well-connected nodes.",
            "based_on_papers": ["Modeling Polypharmacy Side Effects with Graph Convolutional Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of graph attention networks against graph convolutional networks under matched training data and multi-relational edge types for drug-drug interaction prediction.",
            "based_on_papers": ["Graph Attention Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: quantitative analysis of how much molecular substructure information contributes beyond interaction network topology alone, isolated via controlled ablation rather than reported only for the full combined model.",
            "based_on_papers": ["Semi-Supervised Classification with Graph Convolutional Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design cold-start-specific training and evaluation protocols that stratify test splits by node degree to better reflect realistic new-drug scenarios",
        "Develop hybrid architectures that adaptively weight molecular substructure signal versus network topology signal depending on a drug's connectivity in the training graph",
        "Extend multi-relational edge modeling to capture mechanism-specific interaction types (pharmacokinetic versus pharmacodynamic) rather than a single aggregated interaction label",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Graph Convolutional Network (GCN)",
            "url": "https://arxiv.org/abs/1609.02907",
            "description": "Spectral-based graph neural network that aggregates neighbor node features via localized convolutions, used as the base architecture for encoding drug interaction network topology.",
            "verified": True,
        },
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Graph neural network that learns attention weights over neighboring nodes during message passing, used to weight the relative importance of interacting drugs and substructures.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Area under the ROC curve (AUROC) for interaction link prediction",
        "Area under the precision-recall curve (AUPRC)",
        "Precision@K for top-ranked predicted interactions",
        "Cold-start performance stratified by node degree",
    ],
    "implementation_steps": [
        "Construct a drug interaction graph from known interaction records, with drug nodes annotated with molecular substructure features (e.g., fingerprints or SMILES-derived embeddings)",
        "Split edges into training, validation, and test sets, including a dedicated cold-start split stratified by node degree",
        "Train GCN and GAT models to predict interaction links, incorporating multi-relational edge types where mechanism-specific labels are available",
        "Compare against similarity-based and matrix factorization baselines on the same graph and splits",
        "Run controlled ablations isolating the contribution of molecular substructure features versus network topology alone",
        "Evaluate AUROC, AUPRC, and Precision@K overall and separately on the cold-start test split",
    ],
    "unverified_suggestions": [
        "DrugBank drug interaction database",
        "TWOSIDES polypharmacy side effect dataset",
        "BIOSNAP drug-drug interaction network",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets cold-start generalization, a practically important and frequently underevaluated scenario in drug interaction prediction, rather than relying only on random edge-holdout splits",
        "Combining molecular substructure features with network topology is well-motivated and grounded in established prior work rather than treated as an arbitrary architectural choice",
        "The planned ablation isolating substructure versus topology contribution provides genuine mechanistic insight beyond a single reported accuracy number",
        "Including multi-relational edge modeling reflects a realistic understanding that drug interactions have distinct mechanisms rather than being a single undifferentiated interaction type",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how missing or unreported interactions (true negatives versus simply unobserved) will be distinguished when constructing negative training examples, which is a well-known confound in interaction prediction",
        "Dataset fit is a concern: the whitelisted repository (OpenML) does not host curated drug interaction graphs, so the project depends on the unverified DrugBank and TWOSIDES suggestions, which have differing licensing terms and coverage that may affect graph construction",
        "The plan does not address a genuinely important methodological consideration for this domain: mechanism-specific interaction labels used for multi-relational modeling are themselves derived from heterogeneous, sometimes low-confidence sources, and label noise is not discussed",
    ],
    "missing_components": [
        "A discussion of how negative sampling will address the true-negative versus unobserved-interaction ambiguity in the training graph",
        "A plan for assessing sensitivity of results to label noise in mechanism-specific interaction annotations",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.80,
}

CONFIDENCE_JSON = {"internal_consistency": 0.88, "specificity": 0.83, "completeness": 0.89}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph neural networks for drug-drug interaction prediction, based on the retrieved abstract." for t in titles]
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
