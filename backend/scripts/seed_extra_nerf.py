import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Neural Radiance Fields for 3D Scene Reconstruction from Sparse Camera Views"

PLANNER_JSON = {
    "research_objectives": [
        "Develop and evaluate Neural Radiance Field (NeRF) variants optimized for reconstructing 3D scenes from sparse camera views (roughly 3 to 10 images) rather than the dense multi-view captures NeRF traditionally requires",
        "Investigate regularization strategies (depth supervision, geometric priors, frequency annealing) that mitigate overfitting and floating-artifact failure modes under sparse-view NeRF training",
        "Compare NeRF-based sparse-view reconstruction against explicit representations such as 3D Gaussian Splatting on reconstruction fidelity and training/rendering speed",
        "Assess the role of learned cross-scene priors (generalizable/conditional NeRF) in reducing the number of views required for high-fidelity reconstruction of novel scenes",
    ],
    "research_questions": [
        "How few input views can a regularized NeRF use while maintaining novel-view synthesis quality comparable to dense-view baselines?",
        "Which regularization techniques (depth priors, semantic consistency, frequency annealing) most effectively reduce floater and overfitting artifacts in sparse-view settings?",
        "Do generalizable NeRF architectures pretrained across many scenes outperform per-scene optimized sparse-view NeRFs when only a handful of views are available for a new scene?",
        "What are the reconstruction quality and training/rendering speed tradeoffs between implicit (NeRF) and explicit (3D Gaussian Splatting) sparse-view methods?",
    ],
    "keywords": [
        "neural radiance fields",
        "sparse view 3D reconstruction",
        "novel view synthesis",
        "3D Gaussian splatting",
        "generalizable NeRF",
    ],
    "research_scope": (
        "This research covers Neural Radiance Fields and closely related implicit and explicit radiance-field "
        "representations for static scene reconstruction from sparse RGB camera views, including regularization "
        "strategies and generalizable/prior-based conditioning approaches. It compares implicit NeRF variants "
        "against explicit 3D Gaussian Splatting representations on reconstruction fidelity and computational "
        "efficiency. The scope excludes dynamic or 4D scene reconstruction, LiDAR-fused reconstruction pipelines, "
        "and real-time SLAM-based mapping systems."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Regularization-based sparse-view NeRF methods using depth or geometric priors (e.g., DS-NeRF, RegNeRF) to constrain underdetermined reconstruction",
        "Rise of 3D Gaussian Splatting as a faster, explicit alternative to implicit NeRF for sparse- and few-view reconstruction",
        "Generalizable/conditional NeRFs (e.g., pixelNeRF, IBRNet) that learn cross-scene priors to reduce per-scene view requirements",
        "Use of diffusion-based generative priors to hallucinate plausible geometry and appearance in regions unobserved by the sparse input views",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a systematic comparison of regularization strategies (depth supervision versus frequency annealing versus semantic consistency) under a controlled sweep of input view counts.",
            "based_on_papers": ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: characterization of failure modes such as floaters and geometry collapse specifically at extremely low view counts (3 to 5 views) across diverse scene types.",
            "based_on_papers": ["DS-NeRF: Depth-supervised NeRF: Fewer Views and Faster Training for Free"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: evaluation of how well pretrained generalizable NeRF priors transfer to scene categories or camera configurations not represented in their training distribution.",
            "based_on_papers": ["pixelNeRF: Neural Radiance Fields from One or Few Images"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Build standardized sparse-view benchmarks stratified by view count and scene complexity to fairly compare regularization strategies",
        "Investigate hybrid diffusion-prior-plus-geometric-regularization pipelines for extreme few-shot reconstruction",
        "Study cross-domain generalization of pretrained generalizable NeRF and Gaussian Splatting priors to unseen scene categories",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "Neural Radiance Fields (NeRF)",
            "url": "https://arxiv.org/abs/2003.08934",
            "description": "Implicit volumetric scene representation optimized per-scene via differentiable volume rendering; serves as the core sparse-view reconstruction backbone under evaluation.",
            "verified": True,
        },
        {
            "name": "3D Gaussian Splatting",
            "url": "https://arxiv.org/abs/2308.04079",
            "description": "Explicit scene representation using optimizable 3D Gaussians rendered via differentiable rasterization, used as a faster explicit baseline for sparse-view comparison.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Peak Signal-to-Noise Ratio (PSNR)",
        "Structural Similarity Index (SSIM)",
        "Learned Perceptual Image Patch Similarity (LPIPS)",
        "Training and rendering wall-clock time",
    ],
    "implementation_steps": [
        "Assemble sparse-view capture sets (3 to 10 images per scene) with known camera poses from benchmark multi-view datasets, subsampling dense captures to simulate sparsity",
        "Implement a baseline NeRF and augment it with candidate regularization strategies (depth supervision, frequency annealing, semantic consistency losses)",
        "Train a 3D Gaussian Splatting model on the same sparse-view splits as an explicit-representation comparison point",
        "Optionally initialize or condition reconstructions using a pretrained generalizable NeRF prior and fine-tune per scene",
        "Render held-out novel views for each method and compute PSNR, SSIM, and LPIPS against ground-truth images",
        "Profile training and rendering wall-clock time for each method to quantify the quality-versus-speed tradeoff across view counts",
    ],
    "unverified_suggestions": ["DTU MVS dataset", "LLFF dataset", "Mip-NeRF 360 dataset"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal directly targets a well-recognized limitation of NeRF (dense view requirements) with a concrete, measurable sparse-view research question",
        "The methodology includes both implicit (NeRF) and explicit (Gaussian Splatting) representations, enabling a meaningful cross-paradigm comparison rather than a single-architecture study",
        "Evaluation metrics (PSNR, SSIM, LPIPS, wall-clock time) are standard and appropriate for novel-view synthesis quality and efficiency comparison",
        "The identified gaps build logically on cited regularization and generalizable NeRF literature, targeting concrete unaddressed comparisons",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how camera pose estimation error will be controlled for or excluded when comparing methods, which can confound sparse-view reconstruction quality comparisons",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host multi-view 3D reconstruction benchmarks such as DTU or LLFF, so those remain unverified suggestions requiring separate access",
        "The plan does not address how scene complexity (texture-less regions, reflective surfaces) will be stratified in evaluation, despite being a known major source of sparse-view reconstruction failure",
    ],
    "missing_components": [
        "A protocol for controlling or reporting camera pose accuracy across compared methods",
        "An ablation isolating the individual contribution of each regularization term rather than only reporting combined results",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.75,
}

CONFIDENCE_JSON = {"internal_consistency": 0.80, "specificity": 0.78, "completeness": 0.83}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to sparse-view neural radiance field reconstruction, based on the retrieved abstract." for t in titles]
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
