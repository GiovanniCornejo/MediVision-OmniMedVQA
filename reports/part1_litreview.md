# Literature Review: Multimodal Medical Classification

## 1. Introduction

Briefly introduce the scope of the literature review:

- Why multimodal medical classification is important in clinical decision-making.
- The role of vision-language models (VLMs) in combining imaging and textual information.
- What this summary will cover: classification-focused methods, SOTA baselines, and research gaps.

---

## 2. Existing Approaches

### 2.1 Image-only Classification Methods

- CNN/ViT-based models applied to medical imaging classification tasks.
- Strengths: strong visual pattern recognition.
- Limitations: cannot incorporate textual or clinical context, limiting performance on complex cases.

### 2.2 Text-only Classification Methods

- Large language models (LLMs) used for classification from clinical notes or prompts.
- Strengths: captures linguistic patterns.
- Limitations: ignores imaging evidence, may misclassify visually subtle pathologies.

### 2.3 Multimodal Fusion Methods

- Early fusion: combining image and text features before classification.
- Late fusion: combining predictions from separate image and text classifiers.
- Representative works:
  - Gapp et al. (2024): early fusion of X-rays + reports with LLaMA II backbone.
  - Med-Flamingo (2023): few-shot VLM for VQA-style tasks, adapted for classification.
- Advantages: improved accuracy, ability to leverage complementary modalities.
- Limitations: often dataset-specific, may not generalize to multiple domains or unseen tasks.

---

## 3. State-of-the-Art Classification Baselines

- Summarize reported classification metrics on multimodal datasets (e.g., OpenI, FLARE25, MMMED).
- Key observations:
  - VLMs (BiomedCLIP, OpenCLIP, LLaVA, OpenFlamingo) show strong zero-shot/few-shot classification performance.
  - RAG-based models (e.g., MMed-RAG) improve factual accuracy and alignment for clinical tasks.
- Multi-label classification remains challenging due to class imbalance and limited multi-domain datasets.
- Highlight reproducibility: open-source models and datasets vs. proprietary solutions.

---

## 4. Research Gaps in Multimodal Classification

- Single-task focus: most existing methods are evaluated on a single dataset or task.
- Data limitations (imbalances, ambiguous prompts).
- Limited generalizability across multiple imaging modalities (X-ray, ultrasound, microscopy, etc.) and disease categories.
- Interpretability and trustworthiness: many models (especially LLM/VLM-based) are black boxes.
- Resource constraints: large models require significant computational power.
- Integration of multiple tasks: few models can flexibly handle both single-label and multi-label classification across domains.

---

## 5. Conclusion

- Image-only and text-only models are limited for multimodal classification tasks.
- Fusion-based multimodal methods outperform unimodal approaches, but generalization and interpretability challenges remain.
- Motivates your group’s work: developing and benchmarking novel multimodal classifiers for FLARE25, with a focus on both single-label and multi-label classification.

---

## References

[Multimodal Medical Disease Classification with LLaMA II](https://arxiv.org/abs/2412.01306)

[Med-Flamingo: a Multimodal Medical Few-shot Learner](https://proceedings.mlr.press/v225/moor23a)

[On Large Visual Language Models for Medical Imaging Analysis: An Empirical Study](https://ieeexplore.ieee.org/document/10614428)

[MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085)

(Will be auto-populated with `pandoc --citeproc` + `references.bib`.)
