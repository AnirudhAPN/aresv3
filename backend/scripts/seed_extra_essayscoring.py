import json
import os
os.environ["LLM_PROVIDER"] = "api"
import sys
from unittest.mock import patch

sys.path.insert(0, r"c:\Users\aniru\Downloads\ares (2)\aresv3\backend")

TOPIC = "Deep Learning for Automated Essay Scoring and Feedback Generation"

PLANNER_JSON = {
    "research_objectives": [
        "Develop deep learning architectures that predict holistic and trait-specific essay scores with high agreement to expert human raters",
        "Design feedback generation models that produce actionable, rubric-aligned comments on grammar, coherence, argumentation, and organization",
        "Evaluate the generalizability of automated essay scoring (AES) models across prompts, genres, and grade levels",
        "Investigate methods for mitigating scoring bias against non-native English writers and underrepresented student populations",
    ],
    "research_questions": [
        "How do transformer-based models compare to traditional feature-engineered approaches (e.g., handcrafted linguistic features with SVR) for essay scoring accuracy?",
        "What neural architectures best support joint holistic scoring and fine-grained, human-readable feedback generation?",
        "How well do AES models trained on one prompt or dataset transfer to unseen prompts (cross-prompt generalization)?",
        "What techniques can reduce demographic and dialectal bias in automated scoring systems?",
    ],
    "keywords": [
        "automated essay scoring",
        "neural feedback generation",
        "transformer-based text scoring",
        "cross-prompt generalization",
        "writing quality assessment",
    ],
    "research_scope": "This research focuses on deep learning methods for automated essay scoring (AES) and accompanying feedback generation, spanning holistic scoring, trait-level scoring (e.g., organization, word choice, conventions), and natural-language feedback tailored to K-12 and higher-education writing assessment contexts. The scope includes supervised models trained on public corpora such as the ASAP dataset, transformer-based architectures (BERT, RoBERTa) fine-tuned for regression or ordinal classification, and sequence-to-sequence or LLM-based approaches for generating rubric-aligned explanatory feedback. It also considers cross-prompt and cross-domain generalization, fairness and bias mitigation across student subgroups, and evaluation methodologies including quadratic weighted kappa and human-rated feedback usefulness studies. Excluded from scope are non-English essay scoring systems and purely plagiarism-detection tools that do not assess writing quality.",
}

GAP_JSON = {
    "current_trends": [
        "Shift from handcrafted linguistic feature engineering (e.g., readability indices, POS n-grams) toward end-to-end transformer fine-tuning (BERT, RoBERTa) for holistic essay scoring",
        "Growing interest in multi-task models that jointly predict trait-level scores (content, organization, conventions) alongside a holistic score",
        "Emerging use of large language models (LLMs) as zero-shot or few-shot graders paired with prompt-based rubric feedback generation",
        "Increasing attention to fairness auditing, measuring score disparities across native-language background and socioeconomic subgroups",
    ],
    "gaps": [
        {
            "gap_description": "Not addressed in the papers reviewed here: standardized human evaluation protocols for judging the pedagogical usefulness (not just fluency) of generated feedback comments",
            "based_on_papers": ["Automated Essay Scoring with Discourse-Aware Neural Models"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: robustness of AES scoring models to adversarial or gamed essays designed to exploit model shortcuts (e.g., keyword stuffing, length inflation)",
            "based_on_papers": ["A Neural Approach to Automated Essay Scoring", "Cross-Prompt Automated Essay Scoring via Domain Adaptation"],
            "confidence_note": "based on 8 retrieved papers",
        },
        {
            "gap_description": "Not addressed in the papers reviewed here: longitudinal studies measuring whether AES-generated feedback improves student writing outcomes over multiple drafting cycles, as opposed to single-shot scoring accuracy",
            "based_on_papers": ["Neural Automated Essay Scoring and Coherence Modeling"],
            "confidence_note": "based on 8 retrieved papers",
        },
    ],
    "future_directions": [
        "Develop benchmark suites for adversarial robustness testing of AES models against gaming strategies",
        "Design human-in-the-loop evaluation frameworks that measure downstream writing improvement from generated feedback, not just rater agreement",
        "Extend trait-level multi-task scoring architectures to explicitly model argumentation structure and evidence quality",
    ],
    "low_confidence_gaps": False,
}

METHODOLOGY_JSON = {
    "datasets": [
        {
            "name": "HuggingFace Datasets",
            "url": "https://huggingface.co/datasets",
            "description": "Large hub of NLP, vision, and multimodal datasets with a standard loading API.",
            "verified": True,
        }
    ],
    "algorithms": [
        {
            "name": "BERT",
            "url": "https://arxiv.org/abs/1810.04805",
            "description": "Bidirectional transformer encoder fine-tuned with a regression or ordinal-classification head on the [CLS] token to predict holistic or trait-level essay scores from raw essay text.",
            "verified": True,
        },
        {
            "name": "Hierarchical LSTM with Attention (Two-Level RNN)",
            "url": "https://arxiv.org/abs/1606.04289",
            "description": "Sentence-level LSTM encoders feed into a document-level LSTM with attention pooling, producing a score plus attention weights that can be surfaced as coarse feedback highlighting influential sentences.",
            "verified": True,
        },
    ],
    "evaluation_metrics": [
        "Quadratic Weighted Kappa (QWK) between predicted and human-assigned scores",
        "Pearson correlation coefficient between model scores and rater scores",
        "Mean Absolute Error (MAE) on the essay score scale",
        "Human-rated feedback usefulness score (Likert-scale survey of teachers/students on generated feedback comments)",
    ],
    "implementation_steps": [
        "Collect and preprocess essay corpora (e.g., ASAP AES dataset), normalizing prompts, anonymizing student identifiers, and splitting by prompt for cross-prompt evaluation",
        "Tokenize essays and fine-tune a pretrained transformer (BERT/RoBERTa) with a regression head, using prompt-specific score normalization to a common [0,1] range",
        "Train a hierarchical attention baseline (word-to-sentence-to-document LSTM) for comparison and to extract interpretable attention weights for feedback",
        "Implement a feedback generation module (sequence-to-sequence or LLM prompting) conditioned on essay text and predicted trait scores to produce rubric-aligned comments",
        "Evaluate scoring agreement (QWK, Pearson, MAE) in both in-prompt and cross-prompt (leave-one-prompt-out) settings",
        "Run a fairness audit comparing score distributions and error rates across student subgroups (e.g., English language learners vs. native speakers) and conduct a small human study rating feedback usefulness",
    ],
    "unverified_suggestions": [
        "Consider using GPT-4-class LLMs as a zero-shot feedback generator and distilling their outputs into a smaller student model for deployment efficiency",
        "Explore contrastive pretraining on essay pairs of differing quality to improve the model's sensitivity to subtle writing-quality differences",
        "Investigate adversarial training with gamed/adversarial essay examples to improve robustness before deployment in high-stakes settings",
    ],
}

REVIEWER_JSON = {
    "strengths": [
        "Clearly scoped research objectives that connect scoring accuracy, feedback generation, and fairness under one coherent research agenda",
        "Methodology contrasts a modern transformer baseline (BERT) with an interpretable hierarchical attention model, enabling both accuracy and explainability comparisons",
        "Evaluation plan includes both automatic metrics (QWK, Pearson, MAE) and a human-centered usefulness study, which is appropriate for a task with pedagogical stakes",
        "Gap analysis identifies a concrete, underexplored angle (adversarial robustness and longitudinal learning impact) rather than restating well-known scoring-accuracy gaps",
    ],
    "weaknesses": [
        "The methodology does not specify how cross-prompt generalization will be handled when prompts have very different scoring rubrics, which may confound the leave-one-prompt-out evaluation",
        "Evidence sufficiency is limited: only 8 papers were retrieved, which may not adequately cover the most recent LLM-based zero-shot grading literature",
        "No baseline comparison against simple non-neural feature-based models (e.g., readability + SVR) is planned, making it harder to quantify the deep learning models' true gains",
        "The fairness audit step is underspecified regarding which demographic labels are available in the chosen dataset and how they will be obtained ethically",
    ],
    "missing_components": [
        "A data governance / privacy plan for handling student-identifiable writing samples used in fairness auditing",
        "A concrete sample-size and statistical power justification for the human feedback-usefulness study",
    ],
    "novelty_score": 0.68,
    "quality_score": 0.74,
}

CONFIDENCE_JSON = {"internal_consistency": 0.83, "specificity": 0.80, "completeness": 0.85}


def mock_generate(prompt, temperature=0.4, agent=None, **kwargs):
    if "Planner Agent" in prompt:
        return json.dumps(PLANNER_JSON)
    if "Literature Agent" in prompt:
        import re
        titles = re.findall(r'"title":\s*"([^"]*)"', prompt)
        summaries = [f"Discusses {t.rstrip('.')} and its relevance to deep learning approaches for automated essay scoring and feedback generation, based on the retrieved abstract." for t in titles]
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
