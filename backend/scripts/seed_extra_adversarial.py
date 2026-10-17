import json
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Adversarial Robustness of Deep Neural Networks in Autonomous Perception Systems"

PLANNER_JSON = {
    "research_objectives": [
        "Characterize the vulnerability of deep neural network perception models (object detection, semantic segmentation, depth estimation) used in autonomous vehicles to adversarial perturbations under realistic driving conditions.",
        "Evaluate the effectiveness of existing adversarial defense techniques (adversarial training, input transformation, certified defenses) when applied to real-time perception pipelines.",
        "Investigate the transferability of adversarial attacks across different sensor modalities (camera, LiDAR, radar) and fusion architectures.",
        "Develop and benchmark robustness evaluation protocols that reflect physical-world attack scenarios such as adversarial patches and printed perturbations.",
    ],
    "research_questions": [
        "How do white-box and black-box adversarial attacks degrade the accuracy of DNN-based object detectors used in autonomous perception stacks?",
        "Which defense mechanisms provide the best trade-off between robustness gains and inference-time latency for safety-critical perception systems?",
        "To what extent do adversarial perturbations crafted for camera-based models transfer to multi-sensor fusion pipelines involving LiDAR and radar?",
        "What physical-world attack conditions (lighting, weather, viewing angle) most significantly affect the success rate of adversarial patch attacks on autonomous perception?",
    ],
    "keywords": [
        "adversarial robustness",
        "autonomous perception",
        "adversarial patch attacks",
        "adversarial training",
        "sensor fusion robustness",
    ],
    "research_scope": "This research focuses on adversarial robustness of deep neural network perception components (object detection, segmentation, and depth estimation) deployed in autonomous driving and robotics systems, spanning both digital (pixel-level perturbation) and physical (patch, sticker, projection) attack vectors. The scope includes evaluation of white-box, black-box, and transfer attacks along with corresponding defenses such as adversarial training, randomized smoothing, and input preprocessing. It excludes non-vision perception tasks such as pure LiDAR point-cloud classification without sensor fusion, adversarial robustness in non-safety-critical domains (e.g., recommendation systems), and attacks targeting the planning or control layers rather than the perception layer.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing use of adversarial training and its variants (e.g., PGD-based, TRADES) as the de facto standard for hardening perception models, though often benchmarked on classification rather than detection tasks.",
        "Growing interest in physically realizable attacks such as adversarial patches, camouflage textures, and laser/light projection attacks that operate directly in the 3D driving environment.",
        "Emergence of multi-sensor fusion architectures (camera-LiDAR, camera-radar) proposed as a mitigation strategy against single-modality adversarial attacks, with mixed empirical results.",
        "Rising adoption of certified robustness methods (randomized smoothing, interval bound propagation) though largely confined to small-scale image classifiers rather than full perception stacks.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic evaluation of adversarial robustness under combined physical-world nuisance factors (weather, low light, motion blur) alongside adversarial perturbations for object detection models.",
            "based_on_papers": ["Robust Physical-World Attacks on Deep Learning Visual Classification"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized cross-modality transferability benchmarks that quantify how camera-based adversarial perturbations propagate through late-fusion versus early-fusion LiDAR-camera architectures.",
            "based_on_papers": ["Adversarial Examples for Semantic Segmentation and Object Detection", "Towards Deep Learning Models Resistant to Adversarial Attacks"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: real-time latency and computational overhead analysis of certified defense mechanisms when deployed on embedded automotive-grade hardware.",
            "based_on_papers": ["Certified Adversarial Robustness via Randomized Smoothing"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop unified benchmarks that jointly evaluate adversarial robustness and robustness to natural corruptions (weather, sensor noise) for autonomous perception systems.",
        "Design lightweight certified defense mechanisms compatible with the latency constraints of real-time embedded perception hardware.",
        "Extend adversarial robustness evaluation frameworks to multi-modal sensor fusion pipelines with explicit cross-modality transfer attack testing.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "OpenML",
            "url": "https://www.openml.org/",
            "description": "Searchable repository including relevant benchmark datasets with standardized metadata.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN) Object Detector (Faster R-CNN)",
            "url": "https://arxiv.org/abs/1506.01497",
            "description": "A region-proposal-based CNN object detection architecture widely used as the backbone perception model for evaluating adversarial vulnerability in autonomous driving scenarios.",
            "verified": True,
        },
        {
            "name": "Projected Gradient Descent (PGD) Adversarial Training",
            "url": "https://arxiv.org/abs/1706.06083",
            "description": "An iterative min-max adversarial training procedure that hardens neural networks against worst-case bounded perturbations and serves as the standard baseline defense for robustness evaluation.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Mean Average Precision (mAP) under adversarial perturbation",
        "Attack Success Rate (ASR)",
        "Robust Accuracy at fixed perturbation budget (epsilon)",
        "Inference Latency Overhead (ms) under defense mechanisms",
    ],
    "implementation_steps": [
        "Assemble a benchmark suite of camera and LiDAR-camera fusion datasets representative of autonomous driving scenes, including adverse weather and lighting subsets.",
        "Train baseline object detection and segmentation models to convergence on clean data to establish reference performance.",
        "Generate digital adversarial examples using white-box (PGD, FGSM) and black-box (transfer-based, query-based) attack methods against the baseline models.",
        "Fabricate and test physically realizable adversarial patches under varying real-world conditions such as distance, angle, and illumination.",
        "Apply adversarial training and certified defense methods to the baseline models and re-evaluate robustness against the same attack suite.",
        "Analyze cross-modality transferability by measuring attack success rates when perturbations optimized on camera input are propagated through sensor fusion pipelines.",
    ],
    "unverified_suggestions": ["CARLA Simulator", "nuScenes Dataset", "Foolbox Attack Library"],
}

REVIEWER_JSON = {
    "strengths": [
        "The proposal addresses a safety-critical and timely problem with clear real-world implications for autonomous vehicle deployment.",
        "The research questions are well-scoped, distinguishing between digital and physical attack vectors, which is often conflated in prior work.",
        "The methodology combines both attack generation and defense evaluation, providing a balanced empirical framework rather than a one-sided study.",
        "Inclusion of cross-modality transferability analysis addresses an underexplored area relevant to modern multi-sensor autonomous systems.",
    ],
    "weaknesses": [
        "The evidence base of 8 retrieved papers is relatively small for a topic this broad, so conclusions about current trends and gaps should be treated as preliminary rather than exhaustive.",
        "The proposal does not fully specify how physical-world patch fabrication and testing will be controlled for confounding environmental variables.",
        "Dataset fit is uncertain since only OpenML is verified in the whitelist, while more domain-appropriate datasets (e.g., driving-scene benchmarks) remain unverified suggestions.",
        "The proposal lacks a clear discussion of computational resource requirements for adversarial training at scale on detection and segmentation models.",
    ],
    "missing_components": [
        "A discussion of ethical and safety considerations around publishing physically realizable attacks against autonomous vehicle perception systems.",
        "A concrete timeline or phased milestone plan for moving from digital attack evaluation to physical-world validation.",
    ],
    "novelty_score": 0.72,
    "quality_score": 0.76,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.8, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to adversarial robustness in autonomous perception, based on the retrieved abstract." for t in titles]
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
