import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Cross-Modal Retrieval Using Vision-Language Models for E-Commerce Search"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a cross-modal retrieval architecture that maps product images and textual queries into a shared embedding space for e-commerce search.",
        "Evaluate the retrieval accuracy of pretrained vision-language models (e.g., CLIP-style dual encoders) against traditional keyword-based product search baselines.",
        "Investigate the impact of fine-tuning on domain-specific product catalogs with noisy titles, multilingual descriptions, and attribute metadata.",
        "Assess the latency and scalability trade-offs of cross-modal retrieval systems when deployed at production scale in online marketplaces.",
    ],
    "research_questions": [
        "How effectively do vision-language models trained on general web-scale image-text pairs transfer to the e-commerce product search domain?",
        "What fine-tuning strategies (contrastive, hard-negative mining, attribute-conditioned losses) most improve cross-modal retrieval quality on product catalogs?",
        "How do cross-modal retrieval systems compare to unimodal text-based or image-based search in terms of recall, precision, and user-perceived relevance?",
        "What are the computational and latency constraints of deploying large vision-language encoders in real-time e-commerce search pipelines?",
    ],
    "keywords": [
        "cross-modal retrieval",
        "vision-language models",
        "e-commerce search",
        "contrastive image-text learning",
        "product embedding search",
    ],
    "research_scope": (
        "This research focuses on the application of vision-language models to cross-modal retrieval "
        "in e-commerce search, where users query with text, images, or a combination of both to find "
        "relevant products. The scope includes evaluating pretrained dual-encoder architectures such as "
        "CLIP-style models, examining fine-tuning approaches suited to noisy and heterogeneous product "
        "catalog data, and analyzing trade-offs between retrieval accuracy and system latency in "
        "production-scale online shopping platforms. The study does not extend to generative product "
        "recommendation or conversational shopping assistants, focusing instead on the retrieval "
        "component of the search stack."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Widespread adoption of dual-encoder vision-language architectures (e.g., CLIP, ALIGN) for general-purpose image-text retrieval, pretrained on large web-scale caption datasets.",
        "Growing use of contrastive learning objectives to align visual and textual embeddings in a shared latent space for downstream retrieval tasks.",
        "Increasing interest in domain adaptation and fine-tuning of foundation vision-language models for specialized retrieval domains such as fashion and product search.",
        "Emergence of hybrid retrieval pipelines that combine dense cross-modal embeddings with traditional inverted-index keyword search for improved recall.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: the effect of noisy, inconsistent, or auto-generated product titles and descriptions on cross-modal alignment quality in real e-commerce catalogs.",
            "based_on_papers": ["Learning Transferable Visual Models From Natural Language Supervision"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of latency and memory footprint trade-offs when deploying large vision-language dual encoders for real-time, high-QPS product search at marketplace scale.",
            "based_on_papers": ["Attention Is All You Need", "Learning Transferable Visual Models From Natural Language Supervision"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: how fine-grained product attributes (color, size, brand, material) can be explicitly incorporated into cross-modal embedding objectives rather than relying solely on holistic image-caption alignment.",
            "based_on_papers": ["Learning Transferable Visual Models From Natural Language Supervision"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Design attribute-aware contrastive objectives that jointly align product images, titles, and structured metadata fields to improve fine-grained retrieval precision.",
        "Develop lightweight distillation or quantization techniques to reduce the inference cost of vision-language encoders for real-time e-commerce search at scale.",
        "Build benchmark datasets with realistic, noisy e-commerce catalog data to better evaluate cross-modal retrieval robustness under production conditions.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "COCO (Common Objects in Context)", "url": "https://cocodataset.org/", "description": "Object detection, segmentation, and captioning benchmark dataset.", "verified": True}],
    "algorithms": [
        {
            "name": "Transformer",
            "url": "https://arxiv.org/abs/1706.03762",
            "description": "Attention-based sequence architecture used as the backbone for both the text and image encoders in modern vision-language retrieval models.",
            "verified": True,
        },
        {
            "name": "CLIP (Contrastive Language-Image Pretraining)",
            "url": "https://arxiv.org/abs/2103.00020",
            "description": "Dual-encoder vision-language model trained with a contrastive objective to align image and text embeddings in a shared space, forming the basis for cross-modal product retrieval.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Recall@K (K=1,5,10) for text-to-image and image-to-text product retrieval",
        "Mean Reciprocal Rank (MRR)",
        "Normalized Discounted Cumulative Gain (NDCG@10) for ranked search relevance",
        "Query latency (p50/p95) under production-scale load",
    ],
    "implementation_steps": [
        "Collect and preprocess a product catalog dataset including images, titles, descriptions, and structured attribute metadata",
        "Fine-tune a pretrained vision-language dual encoder (e.g., CLIP) using contrastive loss with in-batch and hard-negative product pairs",
        "Build a dense vector index (e.g., approximate nearest neighbor search) over the learned product embeddings for efficient retrieval",
        "Implement a hybrid retrieval pipeline that combines cross-modal embedding similarity with keyword-based filtering for recall boosting",
        "Evaluate retrieval quality offline using Recall@K, MRR, and NDCG on held-out query-product pairs",
        "Benchmark end-to-end query latency and throughput under simulated production traffic to assess deployment feasibility",
    ],
    "unverified_suggestions": [
        "Explore attribute-conditioned contrastive losses that explicitly weight color, size, and brand mismatches during training",
        "Investigate knowledge distillation from large vision-language teacher models into smaller student encoders for low-latency serving",
        "Assess the impact of multilingual product descriptions on cross-modal alignment quality across different marketplace locales",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research objectives that connect foundation vision-language models to a concrete, high-value e-commerce search application.",
        "Methodology grounds the proposal in well-established architectures (Transformer, CLIP) with verifiable, citable sources.",
        "Evaluation plan includes both retrieval-quality metrics (Recall@K, MRR, NDCG) and system-level latency metrics, reflecting production concerns.",
        "Identified gaps around noisy catalog data and attribute-aware alignment are specific and actionable rather than generic.",
    ],
    "weaknesses": [
        "The proposal does not specify how ground-truth relevance judgments for e-commerce query-product pairs would be collected or validated.",
        "Limited discussion of cold-start scenarios for newly listed products with sparse image or text metadata.",
        "The evidence base is relatively narrow, drawing primarily on two foundational papers (Transformer, CLIP) rather than a broader survey of e-commerce-specific retrieval literature.",
        "No baseline comparison against existing production search systems (e.g., BM25 or learned sparse retrieval) is explicitly planned, making relative improvement claims harder to substantiate.",
    ],
    "missing_components": [
        "A concrete data collection and annotation protocol for building or sourcing a labeled e-commerce query-product relevance dataset.",
        "An ablation plan isolating the contribution of attribute-conditioned losses versus standard contrastive fine-tuning.",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.80, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to cross-modal product retrieval, based on the retrieved abstract." for t in titles]
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
