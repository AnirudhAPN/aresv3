import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Predicting Protein-Ligand Binding Affinity in Drug Discovery"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate the accuracy of deep learning architectures for predicting protein-ligand binding affinity compared to traditional scoring functions",
        "Identify structural and molecular representation choices (graph-based, 3D voxel, sequence-based) that most improve predictive performance",
        "Assess the generalization of trained models across novel protein targets and chemically diverse ligand scaffolds",
        "Determine the practical utility of deep learning binding affinity predictors for accelerating early-stage drug discovery pipelines",
    ],
    "research_questions": [
        "How do graph neural networks compare to convolutional and transformer-based models in predicting binding affinity from protein-ligand complexes?",
        "What molecular and protein representations yield the best trade-off between predictive accuracy and computational cost?",
        "How well do current models generalize to unseen protein families and out-of-distribution ligand chemotypes?",
        "What are the main sources of prediction error, and how can uncertainty estimation improve model reliability in virtual screening?",
    ],
    "keywords": [
        "protein-ligand binding affinity",
        "deep learning drug discovery",
        "graph neural networks",
        "molecular docking scoring functions",
        "virtual screening",
    ],
    "research_scope": (
        "This research focuses on deep learning approaches for predicting the binding affinity between "
        "small-molecule ligands and protein targets, a central task in structure-based drug discovery. "
        "The scope spans representation learning methods including graph neural networks, 3D convolutional "
        "models, and sequence/SMILES-based transformers, evaluated primarily on public benchmark datasets "
        "such as PDBbind, BindingDB, and DAVIS/KIBA. The study emphasizes model generalization across novel "
        "protein-ligand pairs, comparison against classical physics-based and empirical scoring functions, "
        "and practical considerations for deployment in high-throughput virtual screening pipelines."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of graph neural networks that operate directly on 3D protein-ligand complex structures rather than hand-crafted descriptors",
        "Growing use of pretrained molecular and protein language models (e.g., SMILES transformers, ESM-based protein embeddings) as feature extractors for affinity prediction",
        "Shift toward multi-task and transfer learning frameworks that combine binding affinity prediction with related tasks such as pose scoring and druggability estimation",
        "Rising interest in uncertainty quantification and ensemble methods to flag low-confidence predictions before use in virtual screening",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of model performance under strict temporal or scaffold-based splits that reflect realistic prospective drug discovery scenarios rather than random cross-validation.",
            "based_on_papers": ["PDBbind: A comprehensive database of experimentally measured binding affinities"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: interpretability methods that connect predicted affinity values to specific structural interactions (e.g., hydrogen bonds, hydrophobic contacts) in a way that is validated against experimental mutagenesis data.",
            "based_on_papers": ["Graph neural networks for predicting protein-ligand binding affinity"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: benchmarking of computational efficiency and inference latency trade-offs across model architectures when applied to billion-scale virtual screening libraries.",
            "based_on_papers": ["DeepDTA: deep drug-target binding affinity prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized scaffold-split and time-split benchmarks to better estimate prospective generalization performance",
        "Integrate physics-informed constraints or free-energy perturbation-derived labels to improve extrapolation to novel chemotypes",
        "Build lightweight, uncertainty-aware model variants suitable for real-time triage in large-scale virtual screening campaigns",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository of ML datasets with standardized benchmarking metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Graph Neural Network (GNN)",
            "url": "https://arxiv.org/abs/1810.00826",
            "description": "Message-passing graph neural network architecture used to learn representations of protein-ligand interaction graphs directly from atomic structure for affinity regression.",
            "verified": True,
        },
        {
            "name": "Transformer-based Sequence Model",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Self-attention transformer architecture applied to SMILES and protein sequence embeddings to capture long-range dependencies relevant to binding affinity prediction.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Root Mean Squared Error (RMSE) between predicted and experimental binding affinity",
        "Pearson correlation coefficient (R) between predicted and true affinity values",
        "Concordance Index (CI) for ranking consistency of predicted affinities",
        "Area Under the ROC Curve (AUROC) for binary active/inactive classification derived from affinity thresholds",
    ],
    "implementation_steps": [
        "Curate and preprocess protein-ligand complex data from PDBbind, BindingDB, and DAVIS/KIBA, removing duplicates and low-resolution structures",
        "Generate molecular graph and 3D structural representations for ligands and protein binding pockets",
        "Split data using scaffold-based and temporal splitting strategies to simulate prospective generalization",
        "Train candidate architectures (GNN, 3D-CNN, transformer) with consistent hyperparameter tuning protocols",
        "Evaluate models on held-out test sets using RMSE, Pearson R, and Concordance Index, with statistical significance testing",
        "Benchmark inference latency and memory footprint for candidate models on large-scale virtual screening libraries",
    ],
    "unverified_suggestions": [
        "Incorporating molecular dynamics-derived conformational ensembles as additional model input to capture binding pocket flexibility",
        "Using active learning to iteratively select the most informative experimental affinity measurements for retraining",
        "Applying contrastive pretraining on unlabeled protein-ligand co-crystal structures before supervised fine-tuning",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly grounds the research questions in well-established public benchmark datasets (PDBbind, BindingDB, DAVIS/KIBA)",
        "The methodology compares multiple representation learning paradigms (graph, 3D, sequence-based) rather than committing to a single architecture",
        "Evaluation metrics are appropriate and standard for the binding affinity prediction literature, enabling comparison with prior work",
        "The identified gaps around scaffold-based splitting and interpretability address genuine, actionable weaknesses in current practice",
    ],
    "weaknesses": [
        "The proposal does not fully justify why scaffold-split evaluation would change conclusions relative to the random splits used in most cited papers, given the limited evidence base of 8 retrieved papers",
        "There is limited discussion of how class imbalance in experimentally measured affinities (skew toward moderate-affinity binders) might bias model training",
        "The plan does not specify how protein structural quality (e.g., resolution, missing residues) will be controlled for across datasets",
        "Computational cost estimates for 3D structural models are not quantified, making feasibility on billion-scale libraries uncertain",
    ],
    "missing_components": [
        "A concrete baseline comparison against classical physics-based scoring functions (e.g., AutoDock Vina, Glide) to contextualize deep learning gains",
        "An explicit data leakage audit protocol for ensuring no ligand or protein overlap between training and test splits",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning-based protein-ligand binding affinity prediction, based on the retrieved abstract." for t in titles]
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
