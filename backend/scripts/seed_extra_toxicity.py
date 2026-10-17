import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Graph Attention Networks for Molecular Toxicity Prediction"

PLANNER_JSON = {
    "research_objectives": [
        "Develop graph attention network (GAT) architectures that represent molecules as atom-bond graphs and learn attention-weighted substructure representations predictive of toxicological endpoints.",
        "Evaluate the ability of multi-head graph attention mechanisms to identify toxicophores and structural alerts without relying on hand-engineered molecular fingerprints or expert-curated substructure rules.",
        "Compare graph attention networks against graph convolutional networks, message-passing neural networks, and traditional fingerprint-based classifiers (e.g., random forest on ECFP) across multiple toxicity endpoints.",
        "Assess the interpretability of learned attention weights as a means of highlighting chemically meaningful substructures relevant to acute toxicity, mutagenicity, and organ-specific toxicity predictions.",
    ],
    "research_questions": [
        "Do graph attention networks achieve higher predictive accuracy than graph convolutional networks and fingerprint-based baselines on standard toxicity benchmarks such as Tox21 and ToxCast?",
        "Can attention weights learned by GATs be reliably interpreted as chemically meaningful toxicophores, and do they align with known structural alerts used in expert toxicology?",
        "How robust are GAT-based toxicity predictors to distributional shift between training chemical libraries and structurally novel compounds encountered at deployment time?",
        "What is the effect of multi-task learning across correlated toxicity endpoints (e.g., multiple nuclear receptor and stress-response assays in Tox21) on the predictive performance of graph attention architectures?",
    ],
    "keywords": [
        "graph attention network",
        "molecular toxicity prediction",
        "graph neural network drug discovery",
        "toxicophore identification",
        "Tox21 benchmark",
    ],
    "research_scope": (
        "This research focuses on the application of graph attention networks (GATs) to molecular property "
        "prediction tasks specific to toxicology, using molecule graphs constructed from SMILES representations "
        "where atoms are nodes and bonds are edges. The scope covers single-task and multi-task GAT architectures "
        "evaluated on standard cheminformatics toxicity benchmarks (Tox21, ToxCast, ClinTox, and related MoleculeNet "
        "datasets), attention-based interpretability methods for toxicophore identification, and comparisons against "
        "graph convolutional networks, message-passing neural networks, and fingerprint-based machine learning "
        "baselines. It excludes de novo molecule generation and inverse design, in vivo or clinical toxicology "
        "studies, quantum-mechanical property prediction unrelated to toxicity, and non-graph-based deep learning "
        "architectures such as pure sequence-based SMILES transformers, which are treated as out of scope for this "
        "review."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of multi-head graph attention layers to weight the contribution of neighboring atoms when constructing molecular embeddings for downstream toxicity classification.",
        "Growing use of multi-task learning setups that jointly predict several correlated Tox21/ToxCast assay endpoints from a shared graph encoder to combat label sparsity in individual assays.",
        "Increasing interest in attention-weight visualization as a lightweight, post-hoc interpretability tool for linking predicted toxicity to specific atoms or substructures within a molecule.",
        "Expansion of benchmark suites (MoleculeNet, Therapeutics Data Commons) that standardize toxicity datasets and splits, enabling more direct comparison between graph attention networks and competing architectures.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: rigorous quantitative validation of attention weights against expert-annotated structural alerts (toxicophores) to confirm that high-attention substructures correspond to chemically established toxicity mechanisms rather than dataset artifacts.",
            "based_on_papers": ["Graph Attention Networks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of GAT toxicity predictors under scaffold-split or time-split evaluation protocols that simulate genuinely novel chemical space, as opposed to random splits that overstate generalization performance.",
            "based_on_papers": ["MoleculeNet: A Benchmark for Molecular Machine Learning", "Analyzing Learned Molecular Representations for Property Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of graph attention networks against message-passing neural networks on identical toxicity endpoints and identical data splits, making it difficult to isolate the specific contribution of the attention mechanism itself.",
            "based_on_papers": ["Neural Message Passing for Quantum Chemistry"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized scaffold-split and time-split benchmark protocols specifically for toxicity prediction to more realistically estimate GAT generalization to structurally novel compounds.",
        "Build quantitative agreement metrics between learned attention weights and established toxicophore libraries (e.g., Derek Nexus alerts) to validate attention-based interpretability claims.",
        "Investigate uncertainty-aware graph attention architectures that flag low-confidence toxicity predictions for compounds far outside the training chemical distribution, supporting safer deployment in early-stage drug screening.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository of ML datasets with standardized benchmarking metadata.", "verified": True}],
    "algorithms": [
        {
            "name": "Graph Attention Network (GAT)",
            "url": "https://arxiv.org/abs/1710.10903",
            "description": "Applies multi-head self-attention over each atom's bonded neighbors to compute learned, non-uniform weights for message aggregation, allowing the model to emphasize atoms and substructures most relevant to a given toxicity endpoint rather than treating all neighbors equally as in standard graph convolution.",
            "verified": True,
        },
        {
            "name": "Message Passing Neural Network (MPNN)",
            "url": "https://arxiv.org/abs/1704.01212",
            "description": "General graph neural network framework that iteratively passes learned messages along bonds and updates atom-level hidden states, used here as a strong non-attention baseline for isolating the specific benefit of attention-weighted aggregation in molecular toxicity prediction.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Area Under the ROC Curve (AUC-ROC) per toxicity endpoint",
        "Area Under the Precision-Recall Curve (AUPRC) for imbalanced assay labels",
        "Macro-averaged F1 score across multi-task toxicity endpoints",
        "Scaffold-split generalization gap (in-distribution vs. out-of-distribution AUC-ROC)",
    ],
    "implementation_steps": [
        "Curate and preprocess molecular toxicity datasets (Tox21, ToxCast, ClinTox) by parsing SMILES strings into atom-bond graphs with standardized atom and bond feature vectors",
        "Generate scaffold-based and random train/validation/test splits to separately assess in-distribution performance and generalization to structurally novel compounds",
        "Implement a multi-head graph attention network with multi-task output heads corresponding to each toxicity assay endpoint, sharing a common graph encoder",
        "Train baseline graph convolutional network and message-passing neural network models on identical splits and features to enable controlled architecture comparisons",
        "Evaluate all models using per-endpoint AUC-ROC, AUPRC, and macro-F1, reporting both random-split and scaffold-split results to quantify the generalization gap",
        "Extract and visualize learned attention weights on held-out toxic compounds, comparing high-attention atoms and substructures against known toxicophore libraries to assess interpretability",
    ],
    "unverified_suggestions": ["Tox21 Data Challenge dataset", "DrugBank chemical structure database", "PubChem BioAssay toxicity records"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal targets a well-motivated, high-impact application: reducing reliance on costly and time-consuming in vivo toxicity screening by predicting toxicity directly from molecular graph structure.",
        "Including both scaffold-split and random-split evaluation is a methodologically sound choice that directly addresses the well-known risk of overstated generalization in molecular property prediction benchmarks.",
        "The multi-task learning setup across correlated Tox21/ToxCast endpoints is a sensible strategy for mitigating label sparsity that commonly affects individual toxicity assays.",
        "Comparing GATs against both graph convolutional and message-passing baselines under controlled, identical splits enables a clean isolation of the attention mechanism's specific contribution.",
    ],
    "weaknesses": [
        "The proposal does not fully specify how attention-weight interpretability will be quantitatively validated against expert toxicophore libraries, leaving evidence sufficiency for interpretability claims unclear beyond qualitative visualization.",
        "Dataset fit is a concern: Tox21 and ToxCast are both derived primarily from in vitro high-throughput screening assays, which may not fully represent the mechanisms underlying organ-level or in vivo toxicity that the broader research motivation references.",
        "There is limited discussion of how class imbalance within individual toxicity assays (where active compounds are often a small minority) will be handled during training beyond the choice of evaluation metric.",
        "The proposal does not address computational scalability of multi-head attention over larger molecules or batches, which could affect feasibility for large-scale virtual screening applications.",
    ],
    "missing_components": [
        "A concrete plan for statistical significance testing when comparing GAT, GCN, and MPNN performance across multiple correlated toxicity endpoints.",
        "An ablation study isolating the effect of the number of attention heads and attention layer depth on both predictive performance and interpretability quality.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to graph attention networks for molecular toxicity prediction, based on the retrieved abstract." for t in titles]
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
