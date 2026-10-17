import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Transformer Models for Automated Radiology Report Generation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop transformer-based architectures that generate clinically accurate free-text radiology reports directly from chest X-ray and CT images",
        "Evaluate cross-modal attention mechanisms that align visual features from radiographic images with medical terminology in generated reports",
        "Investigate methods for reducing factual hallucination and improving clinical correctness in automatically generated radiology findings",
        "Assess the generalizability of transformer-based report generation models across different imaging modalities and patient populations",
    ],
    "research_questions": [
        "How do vision-transformer encoders compare to CNN-transformer hybrid encoders for extracting diagnostically relevant features from radiographic images?",
        "What cross-attention or memory-driven mechanisms best align image regions with corresponding sentences in generated radiology reports?",
        "How can factual consistency and clinical accuracy be measured and improved in transformer-generated radiology reports?",
        "What is the impact of pretraining on large-scale medical image-text corpora on downstream report generation performance?",
    ],
    "keywords": [
        "radiology report generation",
        "vision-language transformers",
        "medical image captioning",
        "chest X-ray interpretation",
        "clinical NLP",
    ],
    "research_scope": "This research focuses on transformer-based deep learning architectures for automated generation of free-text radiology reports from medical images, primarily chest X-rays and CT scans. The scope includes encoder-decoder vision-language transformer designs, cross-modal attention mechanisms, factual consistency and hallucination mitigation techniques, and evaluation using both natural language generation metrics (BLEU, ROUGE) and clinical accuracy metrics (CheXpert labeler F1, clinical efficacy scores). The work is bounded to supervised and self-supervised approaches trained on publicly available paired image-report datasets such as MIMIC-CXR and IU X-Ray, and does not extend to fully unsupervised report generation or non-radiology clinical documentation tasks.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of vision-transformer (ViT) encoders paired with autoregressive transformer decoders for end-to-end report generation",
        "Growing use of memory-augmented and knowledge-graph-conditioned decoders to inject clinical priors and reduce hallucinated findings",
        "Wider reliance on contrastive vision-language pretraining (e.g., CLIP-style objectives on paired chest X-ray and report corpora) before fine-tuning for generation",
        "Shift toward clinically-aware evaluation metrics such as CheXpert-based clinical efficacy scores rather than relying solely on BLEU or ROUGE",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of report generation models on longitudinal studies where prior imaging exams and reports should inform the current report",
            "based_on_papers": ["Generating Radiology Reports via Memory-driven Transformer"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of transformer report generators to distribution shift across scanner vendors, imaging protocols, and patient demographics outside the training hospital system",
            "based_on_papers": ["TieNet: Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized human-in-the-loop protocols for radiologist correction and active learning feedback to iteratively improve deployed report generation models",
            "based_on_papers": ["Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop longitudinal, patient-history-aware transformer architectures that condition report generation on prior studies rather than treating each exam in isolation",
        "Design domain-adaptation and cross-institution benchmarking protocols to quantify and mitigate performance degradation under scanner and population shift",
        "Build clinician-in-the-loop training pipelines that incorporate radiologist edits as reinforcement or preference signals to reduce hallucinated or omitted findings",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "PhysioNet", "url": "https://physionet.org/", "description": "Curated clinical and physiological signal datasets for healthcare research.", "verified": True}
    ],
    "algorithms": [
        {"name": "Transformer", "url": "https://arxiv.org/abs/1706.03762", "description": "Attention-based encoder-decoder architecture used as the backbone decoder for generating free-text radiology report sentences conditioned on encoded image features.", "verified": True},
        {"name": "Vision Transformer (ViT)", "url": "https://arxiv.org/abs/2010.11929", "description": "Patch-based transformer image encoder used to extract global and regional visual features from chest X-ray and CT slices for downstream cross-modal attention with the report decoder.", "verified": True},
    ],
    "evaluation_metrics": [
        "Clinical efficacy F1 score (CheXpert-labeled finding accuracy)",
        "BLEU-4",
        "ROUGE-L",
        "Factual consistency rate (hallucinated vs. confirmed findings)",
    ],
    "implementation_steps": [
        "Assemble and preprocess paired image-report corpora (e.g., MIMIC-CXR, IU X-Ray), normalizing report text and de-identifying patient information",
        "Train or fine-tune a vision-transformer image encoder on the radiographic images to produce patch-level visual embeddings",
        "Pretrain or initialize a transformer decoder with cross-attention layers that attend over the visual embeddings to generate report text autoregressively",
        "Incorporate auxiliary clinical signals such as CheXpert disease labels or knowledge-graph embeddings to condition generation and reduce hallucination",
        "Fine-tune the full encoder-decoder model end-to-end using a combination of cross-entropy loss and reinforcement learning against clinical efficacy reward signals",
        "Evaluate generated reports against radiologist-written ground truth using both NLG metrics and clinical accuracy metrics, followed by qualitative radiologist review",
    ],
    "unverified_suggestions": [
        "Explore retrieval-augmented generation, where similar historical reports are retrieved and used as templates to guide the decoder",
        "Investigate multi-task training that jointly predicts structured findings (e.g., CheXpert labels) alongside free-text report generation",
        "Consider using reinforcement learning from radiologist preference feedback (RLHF-style) to further align generated reports with clinical expectations",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research objectives that connect vision-transformer encoding to clinically meaningful report generation outcomes",
        "Methodology grounds algorithm choices in well-established transformer and vision-transformer architectures with verifiable sources",
        "Evaluation plan appropriately combines standard NLG metrics with clinically-focused metrics like CheXpert-based clinical efficacy",
        "Identified gaps around longitudinal context and cross-institution robustness are specific and actionable for future work",
    ],
    "weaknesses": [
        "The proposal does not fully justify why 8 retrieved papers provide sufficient evidentiary coverage of the rapidly evolving report generation literature",
        "Dataset scope relies heavily on MIMIC-CXR and IU X-Ray, which may not generalize well to the broader clinical population or non-chest imaging modalities",
        "Limited discussion of computational cost and feasibility of training large vision-language transformers on institution-scale hardware",
        "No explicit plan for handling class imbalance between common and rare radiological findings during training",
    ],
    "missing_components": [
        "A concrete plan for radiologist-in-the-loop qualitative evaluation beyond automated metrics",
        "Discussion of data privacy and regulatory considerations for deploying report generation models in clinical settings",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.81, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to transformer-based radiology report generation, based on the retrieved abstract." for t in titles]
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
