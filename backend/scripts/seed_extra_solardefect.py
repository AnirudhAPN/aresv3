import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Convolutional Networks for Automated Solar Panel Defect Detection"

PLANNER_JSON = {
    "research_objectives": [
        "Develop a convolutional neural network capable of automatically detecting and classifying surface defects on photovoltaic solar panels from electroluminescence and RGB imagery.",
        "Evaluate the trade-offs between detection accuracy, inference speed, and model size for deployment on edge devices used in field inspections.",
        "Compare CNN-based defect detection against traditional image processing and manual visual inspection baselines in terms of accuracy and cost.",
        "Investigate the impact of class imbalance and limited labeled defect data on model generalization across different panel manufacturers.",
    ],
    "research_questions": [
        "Which CNN architectures (e.g., ResNet, EfficientNet, custom shallow CNNs) achieve the best accuracy-to-latency ratio for solar panel defect classification?",
        "How does the use of electroluminescence (EL) imaging compare to standard RGB imagery for detecting microcracks and hotspots?",
        "What data augmentation and transfer learning strategies most effectively mitigate the scarcity of labeled defective panel images?",
        "Can lightweight CNNs deployed on drone or robotic inspection platforms achieve real-time defect detection with acceptable accuracy?",
    ],
    "keywords": [
        "solar panel defect detection",
        "convolutional neural networks",
        "electroluminescence imaging",
        "photovoltaic fault classification",
        "automated visual inspection",
    ],
    "research_scope": "This research focuses on the design, training, and evaluation of convolutional neural network architectures for automated detection and classification of physical and electrical defects on solar photovoltaic panels, including microcracks, hotspots, delamination, and soiling. The scope encompasses image acquisition methods such as electroluminescence and infrared thermography, dataset curation and augmentation strategies to address limited labeled defect samples, and comparative benchmarking of CNN architectures against classical computer vision baselines. Deployment considerations for edge and drone-based inspection systems are included, while broader topics such as panel degradation modeling over multi-year timescales and non-CNN defect detection sensors (e.g., acoustic emission) are considered out of scope.",
}

GAP_JSON = {
    "current_trends": [
        "Increasing adoption of electroluminescence and infrared thermography imaging combined with CNNs for non-invasive defect detection in utility-scale solar farms.",
        "Growing use of transfer learning from ImageNet-pretrained backbones (ResNet, VGG, EfficientNet) to compensate for small labeled defect datasets.",
        "Rising interest in lightweight and quantized CNN architectures (MobileNet, SqueezeNet variants) for deployment on drone-mounted or handheld inspection devices.",
        "Emergence of hybrid pipelines combining CNN-based defect classification with unsupervised anomaly detection to flag previously unseen defect types.",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized cross-manufacturer benchmarking of CNN defect detectors, making it difficult to assess whether reported accuracy generalizes beyond the panel types used in training.",
            "based_on_papers": ["Deep Learning-Based Automatic Detection of Photovoltaic Cell Defects", "Electroluminescence Image Analysis for Solar Cell Crack Detection Using CNNs"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness evaluation of CNN defect detectors under real-world field conditions such as variable lighting, dust occlusion, and panel glare during drone-based RGB capture.",
            "based_on_papers": ["Drone-Based Infrared Thermography for Solar Panel Fault Detection"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: systematic cost-benefit analysis comparing automated CNN inspection pipelines against manual thermal inspection in terms of total lifecycle savings for utility-scale solar operators.",
            "based_on_papers": ["A Survey of Machine Learning Approaches for Photovoltaic Fault Diagnosis"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Development of open, multi-manufacturer, multi-imaging-modality benchmark datasets to enable fair comparison of CNN defect detection models.",
        "Integration of self-supervised and few-shot learning techniques to reduce dependency on large labeled defect datasets.",
        "Design of edge-optimized CNN architectures with quantization-aware training for real-time onboard drone inspection without cloud connectivity.",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [{"name": "Kaggle Datasets", "url": "https://www.kaggle.com/datasets", "description": "Community-contributed datasets spanning nearly every domain, with notebooks.", "verified": True}],
    "algorithms": [
        {
            "name": "Convolutional Neural Network (CNN)",
            "url": "https://www.tensorflow.org/tutorials/images/cnn",
            "description": "A deep learning architecture that applies learned convolutional filters to extract hierarchical spatial features, well-suited for classifying microcracks, hotspots, and delamination patterns in solar panel EL and RGB images.",
            "verified": True,
        },
        {
            "name": "Convolutional Autoencoder for Anomaly Detection",
            "url": "https://keras.io/examples/vision/autoencoder/",
            "description": "An unsupervised architecture trained to reconstruct defect-free panel images; reconstruction error highlights unseen or rare defect types not present in labeled training data, complementing supervised CNN classifiers.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Classification accuracy and F1-score per defect category (microcrack, hotspot, delamination, soiling)",
        "Precision-recall AUC to account for class imbalance between defective and non-defective panels",
        "Inference latency (ms per image) on target edge/drone hardware",
        "Intersection-over-Union (IoU) for defect localization when using segmentation or bounding-box outputs",
    ],
    "implementation_steps": [
        "Curate and preprocess a labeled dataset of EL and RGB solar panel images spanning multiple defect categories and panel manufacturers",
        "Apply data augmentation (rotation, brightness jitter, synthetic crack overlay) to address class imbalance and limited defect samples",
        "Fine-tune a pretrained CNN backbone (e.g., ResNet-18 or MobileNetV2) on the augmented dataset using transfer learning",
        "Train a convolutional autoencoder on defect-free panel images to establish an unsupervised anomaly detection baseline",
        "Evaluate both models on a held-out multi-manufacturer test set using accuracy, F1, PR-AUC, and inference latency metrics",
        "Deploy the best-performing lightweight model to a drone or handheld inspection prototype and benchmark real-time field performance",
    ],
    "unverified_suggestions": [
        "Explore vision transformer (ViT) hybrids as a potential accuracy upgrade once labeled dataset size grows beyond typical current benchmarks",
        "Investigate federated learning across multiple solar farm operators to expand training data diversity without centralizing sensitive site imagery",
        "Consider synthetic defect generation via GANs to further augment rare defect classes such as snail trails or bypass diode failures",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "The research objectives are clearly scoped around a well-defined, high-impact industrial inspection problem with measurable outcomes",
        "The methodology appropriately combines supervised CNN classification with unsupervised anomaly detection to handle rare and unseen defect types",
        "Evaluation metrics are well-matched to the class imbalance challenge inherent in defect detection datasets",
        "The proposal grounds architecture choices in established, verifiable transfer learning practices rather than speculative techniques",
    ],
    "weaknesses": [
        "The proposal does not sufficiently address how performance will be validated across panels from different manufacturers and imaging conditions, limiting confidence that findings will generalize",
        "No baseline comparison against traditional (non-CNN) image processing methods such as edge detection or thresholding is specified",
        "The evaluation lacks discussion of statistical significance testing when comparing CNN variants",
        "The proposal does not address computational cost of training on electroluminescence datasets, which are typically much smaller than natural image benchmarks",
    ],
    "missing_components": [
        "A concrete data governance and labeling protocol describing who annotates defects and how inter-annotator agreement is measured",
        "A discussion of failure modes, such as false negatives on safety-critical hotspot defects that could lead to fire risk if missed",
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
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to CNN-based automated solar panel defect detection, based on the retrieved abstract." for t in titles]
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
