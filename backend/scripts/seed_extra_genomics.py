import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Few-Shot Learning for Rare Disease Classification in Genomic Data"

PLANNER_JSON = {
    "research_objectives": [
        "Develop few-shot learning models capable of accurately classifying rare disease phenotypes from genomic and transcriptomic data when only a handful of labeled patient samples are available per condition.",
        "Evaluate meta-learning and metric-learning approaches for their ability to generalize across rare disease categories with extreme class imbalance and limited cohort sizes.",
        "Investigate transfer learning strategies that leverage large-scale common-disease genomic datasets to bootstrap representations useful for rare disease classification.",
        "Assess the interpretability and clinical reliability of few-shot predictions to support genetic counseling and diagnostic decision-making in low-data regimes.",
    ],
    "research_questions": [
        "Which few-shot learning paradigms (metric-based, optimization-based, or hybrid meta-learning) achieve the highest classification accuracy for rare Mendelian and complex genetic disorders given fewer than 20 labeled samples per class?",
        "How does pretraining on large common-disease genomic cohorts (e.g., UK Biobank, gnomAD) affect downstream few-shot performance on rare disease variant classification tasks?",
        "To what extent do few-shot models trained on one genomic data modality (e.g., whole-exome sequencing) transfer to related modalities such as RNA-seq or DNA methylation data?",
        "Can attention-based or prototype-based interpretability methods reliably identify the genomic variants or gene expression signatures driving few-shot rare disease predictions?",
    ],
    "keywords": [
        "few-shot learning genomics",
        "rare disease classification",
        "meta-learning bioinformatics",
        "genomic variant interpretation",
        "low-data clinical genomics",
    ],
    "research_scope": (
        "This research focuses on few-shot and meta-learning approaches applied to the classification of rare "
        "genetic diseases using genomic data modalities including whole-exome/whole-genome sequencing variant "
        "calls, gene expression profiles, and DNA methylation patterns, in settings where fewer than a few dozen "
        "labeled patient samples exist per disease category. The scope includes prototypical networks, "
        "model-agnostic meta-learning (MAML) variants, transfer learning from large common-disease biobanks, and "
        "post-hoc interpretability techniques for identifying causal or contributory genomic features. It excludes "
        "de novo variant calling and sequencing pipeline development, non-genomic rare disease diagnosis (e.g., "
        "imaging-only phenotyping), fully unsupervised clustering of undiagnosed cohorts without any labeled "
        "reference samples, and health-economic or regulatory analyses of rare disease diagnostic deployment, "
        "which are treated as out of scope for this review."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing adoption of prototypical networks and siamese architectures to learn genomic embedding spaces where rare disease samples cluster near class prototypes despite minimal labeled examples.",
        "Increasing use of transfer learning from large-scale population genomic resources such as UK Biobank and gnomAD to pretrain encoders before fine-tuning on rare disease cohorts.",
        "Rising interest in model-agnostic meta-learning (MAML) and its first-order variants to enable rapid adaptation to newly characterized rare disease subtypes with only a few labeled patients.",
        "Expansion of multi-omics integration strategies that combine variant-level, expression-level, and methylation-level features to compensate for the scarcity of labeled rare disease samples in any single modality.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of few-shot model calibration and uncertainty quantification for rare disease predictions, which is critical given the high clinical cost of false negatives in diagnostic settings.",
            "based_on_papers": ["Prototypical Networks for Few-shot Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: direct comparison of meta-learning approaches against simpler baseline strategies such as nearest-centroid classifiers or data augmentation on real multi-cohort rare disease genomic datasets, rather than synthetic few-shot benchmarks derived from common-disease data.",
            "based_on_papers": ["Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "Matching Networks for One Shot Learning"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-institutional validation of few-shot rare disease classifiers to assess robustness to batch effects and sequencing platform differences, which commonly confound genomic transfer learning across sites.",
            "based_on_papers": ["Deep Learning for Genomic Prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop calibrated few-shot classifiers with built-in abstention mechanisms so that low-confidence rare disease predictions are automatically routed to expert genetic counselors rather than reported directly.",
        "Build standardized multi-institutional rare disease genomic benchmarks with harmonized phenotype labels to enable fair comparison of meta-learning methods across sequencing platforms.",
        "Integrate multi-omics few-shot fusion architectures that jointly leverage variant, expression, and methylation data to improve classification robustness when any single modality has extremely few labeled samples.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "MIMIC-III / MIMIC-IV",
            "url": "https://physionet.org/content/mimiciv/",
            "description": "De-identified ICU clinical records, a common healthcare-AI benchmark.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Prototypical Networks (Metric-Based Meta-Learning)",
            "url": "https://arxiv.org/abs/1703.05175",
            "description": "Learns a genomic feature embedding space in which each rare disease class is represented by a prototype (mean embedding of its few labeled samples), enabling classification of new patients via nearest-prototype distance, which is well suited to the extreme class imbalance seen in rare disease cohorts.",
            "verified": True,
        },
        {
            "name": "Model-Agnostic Meta-Learning (MAML)",
            "url": "https://arxiv.org/abs/1703.03400",
            "description": "Optimization-based meta-learning approach that learns an initialization of a genomic classifier's parameters that can be rapidly fine-tuned to a new rare disease category using only a handful of gradient steps and labeled samples, enabling fast adaptation as new rare disease subtypes are characterized.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "N-way K-shot classification accuracy",
        "Area Under the Precision-Recall Curve (AUPRC) per disease class",
        "Macro-averaged F1 score across rare disease categories",
        "Expected Calibration Error (ECE) for predictive confidence",
    ],
    "implementation_steps": [
        "Assemble a multi-cohort rare disease genomic dataset spanning whole-exome/whole-genome variant calls and, where available, transcriptomic profiles, harmonizing phenotype labels to a standard ontology such as Orphanet or HPO",
        "Pretrain a genomic feature encoder on large common-disease or population reference cohorts (e.g., gnomAD, UK Biobank) to learn general-purpose variant and expression representations",
        "Construct episodic few-shot training tasks (N-way K-shot) by sampling small support and query sets from rare disease classes to simulate the low-data regime encountered at deployment",
        "Train and compare metric-based (prototypical networks) and optimization-based (MAML) meta-learning models on the episodic tasks, using the pretrained encoder as a frozen or fine-tunable backbone",
        "Evaluate held-out rare disease classes not seen during meta-training to measure true generalization to novel disease categories, reporting per-class AUPRC and macro-F1",
        "Apply calibration analysis and prototype-distance-based interpretability to assess which genomic features drive classification decisions and whether predicted confidence aligns with observed accuracy",
    ],
    "unverified_suggestions": ["ClinVar rare variant annotations", "Orphanet rare disease registry", "gnomAD population allele frequency database"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets a high-impact, underserved clinical problem, since rare disease diagnosis is frequently delayed for years due to the scarcity of labeled cases for any given condition.",
        "The methodology thoughtfully pairs a metric-based approach (prototypical networks) with an optimization-based approach (MAML), enabling a meaningful comparison of adaptation strategies under genuine few-shot constraints.",
        "Pretraining on large population genomic resources before few-shot fine-tuning is a sound strategy for mitigating the fundamental data scarcity of rare disease research.",
        "The evaluation plan appropriately emphasizes per-class AUPRC and macro-F1 rather than raw accuracy, which is important given the severe class imbalance inherent to rare disease cohorts.",
    ],
    "weaknesses": [
        "The proposal does not fully specify how episodic few-shot tasks will be constructed from genuinely limited rare disease cohorts without leaking information between meta-training and meta-testing splits, leaving evidence sufficiency for the reported generalization claims unclear.",
        "Dataset fit is a concern: pretraining resources such as UK Biobank are demographically skewed and predominantly capture common variants, which may not transfer well to the rare, often population-specific variants underlying many rare diseases.",
        "There is limited discussion of how batch effects and cross-platform sequencing differences between the pretraining cohort and the rare disease cohort will be corrected before few-shot adaptation.",
        "The proposal does not address how the model should behave when a patient's presentation does not match any known rare disease prototype, i.e., true novel-class detection versus forced classification.",
    ],
    "missing_components": [
        "A concrete plan for statistical significance testing when comparing prototypical networks against MAML and simpler baselines across multiple rare disease classes.",
        "An ablation study isolating the contribution of population-cohort pretraining versus training the encoder from scratch on rare disease data alone.",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to few-shot classification of rare diseases from genomic data, based on the retrieved abstract." for t in titles]
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
