import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Capsule Networks for Robust Image Classification Under Viewpoint Variation"

PLANNER_JSON = {
    "research_objectives": [
        "Evaluate how well capsule network architectures preserve part-whole spatial relationships and maintain classification accuracy when input images undergo significant viewpoint or pose variation",
        "Design and benchmark dynamic routing and EM-routing capsule variants against CNN and vision transformer baselines under controlled rotation, translation, and affine viewpoint perturbations",
        "Quantify the extent to which capsule networks generalize to unseen viewpoints not present in training data, using pose-augmented and out-of-distribution test splits",
        "Identify architectural modifications to capsule routing mechanisms that improve robustness and computational efficiency for viewpoint-invariant image classification",
    ],
    "research_questions": [
        "Do capsule networks with dynamic or EM routing outperform CNN and transformer baselines on image classification accuracy under synthetic and natural viewpoint variation?",
        "How does the number of routing iterations affect the tradeoff between viewpoint robustness and training/inference computational cost in capsule networks?",
        "Can capsule networks generalize to novel, unseen viewpoints better than data-augmented CNNs, or does their advantage depend on similar viewpoint coverage during training?",
        "What is the sensitivity of capsule network robustness to the choice of primary capsule dimensionality and routing algorithm across datasets of varying visual complexity?",
    ],
    "keywords": [
        "capsule networks",
        "dynamic routing",
        "viewpoint invariance",
        "pose equivariance",
        "robust image classification",
    ],
    "research_scope": (
        "This research focuses on capsule network architectures, including dynamic routing-by-agreement and "
        "expectation-maximization (EM) routing variants, applied to image classification tasks under viewpoint "
        "and pose variation. It covers evaluation on affine-transformed and multi-view benchmark datasets, "
        "comparisons against CNN and vision transformer baselines with and without data augmentation, and "
        "analysis of routing iteration count, capsule dimensionality, and computational overhead. The scope "
        "excludes capsule-based generative modeling, video or 3D point-cloud applications, and object detection "
        "or segmentation tasks, which are treated as out of scope for this analysis."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Continued refinement of routing-by-agreement algorithms (dynamic routing, EM routing, self-attention routing) aimed at reducing computational cost while preserving pose-awareness",
        "Growing use of affine-transformed and multi-view benchmark datasets (e.g., affNIST, SmallNORB) to stress-test viewpoint robustness beyond standard test sets",
        "Increasing comparison of capsule networks against vision transformers and augmented CNNs, questioning whether capsules provide a genuine viewpoint-generalization advantage over sufficiently augmented baselines",
        "Interest in hybrid architectures that combine convolutional feature extraction with capsule-based pose-vector representations to reduce the parameter and routing overhead of pure capsule networks",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled, unified benchmark isolating viewpoint generalization from mere data augmentation effects, so that capsule networks and CNN/transformer baselines are compared under identical unseen-viewpoint test protocols.",
            "based_on_papers": ["Dynamic Routing Between Capsules"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic analysis of how routing iteration count and primary capsule dimensionality jointly affect the accuracy-robustness-compute tradeoff across datasets of varying visual complexity.",
            "based_on_papers": ["Matrix Capsules with EM Routing", "Dynamic Routing Between Capsules"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of capsule network robustness on natural (non-synthetic) multi-view image datasets with real camera pose variation, as opposed to purely affine-transformed synthetic benchmarks.",
            "based_on_papers": ["Matrix Capsules with EM Routing"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop standardized unseen-viewpoint evaluation protocols that decouple architectural pose-equivariance from data augmentation coverage",
        "Investigate hybrid CNN-capsule architectures that reduce routing computational overhead while retaining part-whole pose representation benefits",
        "Extend robustness evaluation to natural multi-view datasets with real camera pose changes rather than relying solely on synthetic affine perturbations",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "ImageNet",
            "url": "https://www.image-net.org/",
            "description": "Large-scale labeled image dataset for computer vision.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://arxiv.org/abs/1512.03385",
            "description": "Standard deep convolutional architecture (e.g., ResNet) used as a baseline classifier, optionally combined with viewpoint-augmentation training to compare against capsule-based pose representations.",
            "verified": True,
        },
        {
            "name": "Capsule Network with Dynamic Routing",
            "url": "https://arxiv.org/abs/1710.09829",
            "description": "Architecture that encodes visual entities as pose vectors in capsules and uses iterative routing-by-agreement between capsule layers to preserve part-whole spatial relationships under viewpoint change.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Top-1 classification accuracy",
        "Accuracy degradation under unseen viewpoint/affine perturbation",
        "Macro-averaged F1 score",
        "Inference latency and routing computational overhead",
    ],
    "implementation_steps": [
        "Curate and preprocess image datasets with controlled affine and multi-view viewpoint transformations, splitting seen and unseen viewpoint subsets",
        "Implement baseline CNN and vision transformer classifiers with and without viewpoint-focused data augmentation",
        "Implement capsule network variants (dynamic routing and EM routing) with configurable routing iteration count and primary capsule dimensionality",
        "Train all models under matched compute budgets and hyperparameter tuning protocols on the seen-viewpoint training split",
        "Evaluate all models on seen and unseen viewpoint test splits, recording accuracy degradation, F1, and inference latency",
        "Conduct ablations varying routing iterations and capsule dimensionality to characterize the robustness-compute tradeoff",
    ],
    "unverified_suggestions": ["affNIST", "SmallNORB", "MultiPIE"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal clearly isolates viewpoint generalization as a distinct research question from generic classification accuracy, with explicit seen/unseen viewpoint splits",
        "The methodology specifies a concrete, reproducible comparison pipeline across capsule, CNN, and transformer architectures under matched compute budgets",
        "Evaluation metrics appropriately capture both robustness (accuracy degradation under perturbation) and practical deployment cost (inference latency, routing overhead)",
        "The identified gaps build logically on limitations of the cited dynamic routing and EM routing capsule literature",
    ],
    "weaknesses": [
        "The proposal does not fully specify how 'unseen viewpoint' will be operationalized and validated as genuinely distinct from the augmentation distribution used during training",
        "Evidence sufficiency is limited: only a small set of retrieved papers directly benchmark capsule networks against augmented CNN/transformer baselines under identical viewpoint protocols, so conclusions should be treated as preliminary until validated on the proposed unified benchmark",
        "Dataset fit caveats apply since the whitelisted dataset repository (ImageNet) does not natively provide the affine or multi-view pose annotations implied by the unverified suggestions (affNIST, SmallNORB, MultiPIE)",
        "The plan does not address how routing algorithm instability (e.g., sensitivity to initialization) will be controlled for across repeated training runs",
    ],
    "missing_components": [
        "A clear statistical significance testing plan across repeated runs and viewpoint splits",
        "An explicit ablation isolating the effect of capsule pose-vector representation from routing iteration count alone",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to capsule network robustness under viewpoint variation, based on the retrieved abstract." for t in titles]
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
