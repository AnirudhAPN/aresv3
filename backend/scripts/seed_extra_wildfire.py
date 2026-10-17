import os
os.environ["LLM_PROVIDER"] = "api"   # MUST be the very first thing, before any project imports —
# agent code branches to the local model BEFORE calling llm_service when
# LLM_PROVIDER=local, which would silently bypass the mocks below and train
# the model on its own (bad) output instead of this hand-written data.

import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Wildfire Spread Prediction Using Multispectral Satellite Imagery"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a deep learning model that predicts short-term wildfire spread direction and rate using multispectral satellite imagery combined with weather and terrain data",
        "Evaluate convolutional and spatiotemporal (ConvLSTM, 3D-CNN) architectures for capturing fire-front evolution across successive satellite observation timesteps",
        "Investigate the contribution of individual multispectral bands and derived fire indices (e.g., NBR, NDVI) to spread prediction accuracy",
        "Assess model transferability across different biomes and fuel types and its sensitivity to satellite revisit-time gaps",
    ],
    "research_questions": [
        "Can spatiotemporal deep learning models predict next-timestep wildfire perimeter growth more accurately than physics-based cellular automata fire-spread models given the same input data?",
        "Which multispectral bands and derived indices contribute most to predictive accuracy for active fire-spread modeling?",
        "How does prediction accuracy degrade as satellite revisit-time gaps increase, from near-continuous geostationary observation to daily polar-orbiting revisit?",
        "Do models trained on one biome or fuel type generalize to wildfires in different ecosystems without retraining?",
    ],
    "keywords": [
        "wildfire spread prediction deep learning",
        "multispectral satellite imagery",
        "ConvLSTM fire spread",
        "burn severity remote sensing",
        "wildfire perimeter forecasting",
    ],
    "research_scope": (
        "This research covers deep learning models for short-term (hours to one to two days) wildfire spread "
        "and perimeter growth prediction using multispectral satellite imagery such as MODIS, VIIRS, and "
        "Sentinel-2, fused with weather and terrain covariates. It includes spatiotemporal architecture design "
        "and band/index contribution analysis. The scope excludes long-term fire risk or susceptibility mapping, "
        "smoke plume dispersion modeling, and approaches relying solely on ground-based sensor networks without "
        "satellite imagery."
    ),
}

GAP_JSON = {
    "current_trends": [
        "Growing use of ConvLSTM and 3D-CNN architectures to model spatiotemporal fire-front evolution from satellite image sequences",
        "Fusion of multispectral fire indices (NBR, NDVI, active fire detections) with weather (wind, humidity, temperature) and terrain (slope, fuel type) covariates as multi-channel model input",
        "Adoption of graph neural networks and cellular-automata-inspired neural surrogates to model fire spread as a propagating front over terrain graphs",
        "Increasing use of geostationary satellite data (e.g., GOES) for near-continuous active fire monitoring feeding real-time prediction pipelines",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: a controlled comparison of how satellite revisit-time gaps, from geostationary to polar-orbiting cadence, affect deep learning spread-prediction accuracy.",
            "based_on_papers": ["Deep learning for wildfire spread prediction using multi-source remote sensing data"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: cross-biome generalization testing, since most models are trained and evaluated within a single fire-prone region or fuel type.",
            "based_on_papers": ["A convolutional neural network for wildfire spread prediction"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: the individual contribution of specific multispectral bands versus derived composite indices to spread prediction accuracy, since most studies use pre-fixed input band stacks.",
            "based_on_papers": ["Deep learning for wildfire spread prediction using multi-source remote sensing data"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Construct multi-biome benchmark datasets with consistent perimeter labeling to systematically test cross-region generalization",
        "Perform systematic band and index ablation studies to identify minimal sufficient multispectral input sets",
        "Explore neural surrogate models that fuse geostationary and polar-orbiting satellite streams for real-time fire-spread prediction",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {"name": "OpenML", "url": "https://www.openml.org/", "description": "Searchable repository including relevant benchmark datasets with standardized metadata.", "verified": True},
    ],
    "algorithms": [
        {
            "name": "ConvLSTM",
            "url": "https://arxiv.org/abs/1506.04214",
            "description": "Convolutional recurrent architecture that jointly models spatial fire-front patterns and their temporal evolution across sequential satellite observations.",
            "verified": True,
        },
        {
            "name": "U-Net",
            "url": "https://arxiv.org/abs/1505.04597",
            "description": "Encoder-decoder segmentation architecture adapted to predict per-pixel burned-area probability maps for next-timestep fire perimeter estimation.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Intersection over Union (IoU) of predicted versus actual burned area",
        "F1-score for fire/no-fire pixel classification",
        "Mean absolute error of predicted spread rate",
        "Sørensen-Dice coefficient for perimeter overlap",
    ],
    "implementation_steps": [
        "Assemble time-aligned multispectral satellite image sequences (active fire detections, NBR, NDVI bands) with paired weather reanalysis and terrain covariates",
        "Preprocess and co-register imagery across sensors and timesteps, resampling to a consistent spatial grid and temporal cadence",
        "Implement and train a ConvLSTM and a U-Net-based baseline to predict next-timestep burned-area probability maps",
        "Conduct band and index ablation experiments by systematically removing or substituting multispectral input channels",
        "Evaluate spread prediction accuracy using IoU, F1-score, spread-rate MAE, and Dice coefficient on held-out fire events",
        "Test cross-biome generalization by training on one region/fuel type and evaluating zero-shot on fires in a different ecosystem",
    ],
    "unverified_suggestions": ["MODIS Active Fire product", "VIIRS Active Fire data", "Sentinel-2 imagery via Copernicus Open Access Hub"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal targets a clearly defined, high-impact short-term prediction task with concrete, measurable spatial and temporal accuracy metrics",
        "The methodology directly compares deep learning approaches against physics-based cellular automata baselines, grounding claimed improvements in an established reference",
        "Planned band and index ablation studies provide interpretable insight into which multispectral inputs actually drive predictive performance",
        "Cross-biome generalization testing is explicitly built into the evaluation plan rather than treated as an afterthought",
    ],
    "weaknesses": [
        "All 3 claimed gaps rely on a small set of retrieved papers — treat with caution per the evidence-sufficiency rule-check.",
        "The proposal does not specify how missing or cloud-obscured satellite observations, which are common in active fire monitoring, will be handled during training and inference",
        "Dataset fit is uncertain: the whitelisted dataset repository (OpenML) does not typically host satellite fire-monitoring products such as MODIS or VIIRS active fire data, so those remain unverified suggestions requiring separate access via NASA/Copernicus portals",
        "The plan does not address how weather forecast uncertainty, as opposed to weather reanalysis data used for training, would propagate into operational real-time spread prediction",
    ],
    "missing_components": [
        "A cloud/smoke occlusion handling strategy for satellite input gaps during both training and real-time inference",
        "A discussion of how forecast-weather uncertainty would be incorporated for operational (as opposed to retrospective) deployment",
    ],
    "novelty_score": 0.70,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.79, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to wildfire spread prediction from multispectral satellite imagery, based on the retrieved abstract." for t in titles]
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
