import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Knowledge Graph Embeddings for Drug Repurposing Discovery"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate the effectiveness of knowledge graph embedding methods for predicting novel drug-disease associations from biomedical knowledge graphs",
        "Compare translational, semantic-matching, and graph neural network-based embedding architectures on drug repurposing benchmark tasks",
        "Assess how heterogeneous entity and relation types (genes, pathways, side effects, phenotypes) influence the quality of learned drug and disease embeddings",
        "Investigate the interpretability and biological plausibility of top-ranked repurposing candidates generated from embedding-based link prediction",
    ],
    "research_questions": [
        "How do knowledge graph embedding models such as TransE, ComplEx, and R-GCN compare in link prediction accuracy for drug-disease associations?",
        "What impact does incorporating multi-relational biomedical context (e.g., protein-protein interactions, gene expression, pathway membership) have on repurposing candidate ranking?",
        "How well do embedding-based predictions generalize to diseases and drugs that are sparsely represented in the training knowledge graph?",
        "Can attention-based or path-based explanation methods provide biologically interpretable justifications for predicted drug repurposing candidates?",
    ],
    "keywords": [
        "knowledge graph embeddings",
        "drug repurposing",
        "biomedical link prediction",
        "graph neural networks",
        "drug-disease association",
    ],
    "research_scope": (
        "This research focuses on knowledge graph embedding techniques applied to large-scale biomedical "
        "knowledge graphs for the purpose of drug repurposing discovery. The scope covers translational "
        "distance models (e.g., TransE, TransR), semantic matching models (e.g., ComplEx, DistMult), and "
        "graph neural network-based encoders (e.g., R-GCN, CompGCN) trained on heterogeneous graphs "
        "integrating drugs, diseases, genes, proteins, pathways, and side effects sourced from resources "
        "such as Hetionet, DRKG, and DrugBank. The study emphasizes link prediction performance for "
        "novel drug-disease associations, robustness to sparsely connected entities, and the biological "
        "interpretability of predicted repurposing candidates relative to existing clinical and "
        "experimental evidence."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of graph neural network encoders (R-GCN, CompGCN) that jointly model multiple relation types instead of treating each relation independently",
        "Increasing use of large integrated biomedical knowledge graphs such as Hetionet and the Drug Repurposing Knowledge Graph (DRKG) that combine drug, gene, disease, and pathway data from multiple curated sources",
        "Rising interest in combining knowledge graph embeddings with molecular structure or gene expression features in multimodal repurposing pipelines",
        "Emerging focus on explainable link prediction, using attention weights or relation paths to justify why a drug-disease pair was ranked highly",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of embedding model performance specifically on cold-start diseases with few or no existing drug associations in the knowledge graph, which is the most clinically relevant repurposing scenario.",
            "based_on_papers": ["Hetionet: an integrative network of disease and drug repurposing knowledge"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: prospective validation of top-ranked repurposing predictions against real-world clinical trial outcomes or electronic health record signals, rather than retrospective held-out link prediction accuracy alone.",
            "based_on_papers": ["Drug Repurposing Knowledge Graph (DRKG) for Covid-19"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized comparison of knowledge graph construction choices (entity resolution, relation schema design, negative sampling strategy) and their downstream effect on repurposing candidate quality across different embedding architectures.",
            "based_on_papers": ["Translating Embeddings for Modeling Multi-relational Data"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop cold-start evaluation protocols that hold out entire diseases or drug classes to better simulate realistic repurposing discovery conditions",
        "Integrate embedding-based candidate ranking with downstream clinical validation signals such as adverse event databases and trial registries",
        "Standardize knowledge graph construction and negative sampling benchmarks to enable fair comparison across embedding architectures",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "UCI Machine Learning Repository", "url": "https://archive.ics.uci.edu/", "description": "Large collection of classic ML benchmark datasets across domains.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Relational Graph Convolutional Network (R-GCN)",
            "url": "https://arxiv.org/abs/1703.06103",
            "description": "Graph neural network architecture that applies relation-specific weight matrices during message passing, enabling it to model the heterogeneous entity and relation types present in biomedical knowledge graphs.",
            "verified": True,
        },
        {
            "name": "ComplEx Knowledge Graph Embedding",
            "url": "https://arxiv.org/abs/1606.06357",
            "description": "Semantic matching embedding model that represents entities and relations as complex-valued vectors, allowing it to capture both symmetric and antisymmetric relation patterns common in drug-disease and drug-gene links.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Mean Reciprocal Rank (MRR) for drug-disease link prediction",
        "Hits@k (k=1,3,10) for ranking accuracy of held-out associations",
        "Area Under the ROC Curve (AUROC) for binary repurposing candidate classification",
        "Area Under the Precision-Recall Curve (AUPRC) given class imbalance between true and negative drug-disease pairs",
    ],
    "implementation_steps": [
        "Construct a unified biomedical knowledge graph by integrating drug, disease, gene, pathway, and side-effect entities from Hetionet, DRKG, and DrugBank with consistent entity resolution",
        "Generate training, validation, and test splits using both random edge holdout and cold-start disease holdout to evaluate realistic repurposing scenarios",
        "Train candidate embedding models (TransE, ComplEx, R-GCN, CompGCN) under a shared hyperparameter search and negative sampling protocol",
        "Evaluate link prediction performance using MRR, Hits@k, AUROC, and AUPRC on held-out drug-disease association test sets",
        "Rank predicted novel drug-disease links for a target disease set and cross-reference top candidates against ClinicalTrials.gov and existing pharmacological literature",
        "Apply path-based or attention-based explanation methods to surface the biomedical relations supporting each high-confidence repurposing prediction",
    ],
    "unverified_suggestions": [
        "Incorporating gene expression signatures (e.g., LINCS L1000) as auxiliary node features to enrich drug and disease embeddings",
        "Using active learning to prioritize which predicted drug-disease pairs should be sent for experimental or clinical validation first",
        "Applying contrastive pretraining across multiple biomedical knowledge graphs before fine-tuning on the target repurposing task",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal grounds its research questions in well-established, publicly available biomedical knowledge graphs (Hetionet, DRKG, DrugBank)",
        "The methodology compares multiple embedding paradigms (translational, semantic matching, graph neural network) rather than committing to a single architecture",
        "The evaluation protocol appropriately includes cold-start disease holdout in addition to standard random splits, addressing a realistic repurposing scenario",
        "The identified gaps around prospective clinical validation and cold-start evaluation target genuine, actionable weaknesses in current practice",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small, limiting confidence that the identified gaps generalize across the broader drug repurposing literature",
        "The proposal does not fully address how entity resolution errors when merging multiple knowledge graphs (Hetionet, DRKG, DrugBank) could propagate into biased or noisy embeddings",
        "There is limited discussion of how negative sampling strategy choices affect the reliability of AUROC and AUPRC comparisons across embedding models",
        "Computational cost and scalability of graph neural network encoders on knowledge graphs with millions of edges are not quantified, making feasibility uncertain",
    ],
    "missing_components": [
        "A concrete baseline comparison against non-embedding repurposing approaches, such as chemical similarity-based or gene expression signature reversal methods",
        "An explicit protocol for expert or literature-based validation of top-ranked repurposing candidates beyond automated link prediction metrics",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.84, "specificity": 0.81, "completeness": 0.86}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to knowledge graph embedding methods for drug repurposing discovery, based on the retrieved abstract." for t in titles]
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
