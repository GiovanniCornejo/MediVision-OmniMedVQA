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

### Notes on [Multimodal Medical Disease Classification with LLaMA II](https://arxiv.org/abs/2412.01306)

- **Task**
  - Disease classification from chest X-rays and clinical reports (multimodal input).
- **Dataset (OpenI)**
  - 2D chest X-rays ($256 \times 256$) paired with clinical reports.
  - Labels: 14 disease classes (multi-label classification).
    - Includes: _Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, Pneumonia, Pneumothorax, Support-Devices etc._
    - _No Finding_ acts as the mutually exclusive class.
  - Split: 3199 training, 101 validation, 377 test samples.
  - **Imbalance**: _No Finding_ dominates. Many diseases have only a few training samples.
- **Model Design**
  - Backbone: **LLaMA II 7B** (language model)
  - **Text features**
    - Clinical reports tokenized and embedded into 4096-dim vectors with positional embeddings.
  - **Image features**
    - X-rays split into 16 patches ($32 \times 32$).
    - Extracted via 2D convolution, projected into same 4096-dim space as text features.
  - Architecture:
    - Three transformer-based modules (text, vision, fusion).
    - Fusion via cross-layers with three strategies:
      - **Early Fusion** (parallel): text and vision fused at each level.
      - **Late Fusion** (serial): fusion after modality-specific encoders.
      - **Mixed Fusion**: combination of both parallel and serial.
- **Training**
  - Fine-tuning with **LoRA (Low-Rank Adaptation)** to reduce GPU memory cost and computation time while maintaining the same or better performance.
  - Multiple LoRA configurations tested ($r = 2, 4, 8$).
- **Results**
  - Metric: **mean AUC (ROC)**
  - Best:
    - Early Fusion (Parallel, $r=2$): **0.971 AUC**
    - Late Fusion (Serial, $r=2$): **0.967 AUC**
  - Both outperform baseline TransCheX (**0.963 AUC**).
- **Key Contributions**
  - Demonstrates strong performance of transformer-based multimodal fusion for medical classification.
  - Evaluates and compares early, late, and mixed fusion pipelines.
  - Shows LoRA can efficiently fine-tune large models for small, domain-specific datasets.
- **LitReview Relevance**
  - Fits naturally in 2.3 Multimodal Fusion Methods (discussion of fusion strategies in vision-language models).
  - Also relevant to 3. SOTA Classification Methods (demonstrates high AUC on OpenI dataset).
  - Potential angle for 4. Research Gaps:
    - Dataset imbalance (overrepresentation of _No Finding_).
    - Potential for limited generalization due to small dataset size, despite good results.

### Notes on [Med-Flamingo: a Multimodal Medical Few-shot Learner](https://proceedings.mlr.press/v225/moor23a)

- **Task**
  - Generative medical visual question answering (VQA) using interleaved text + image input.
  - Focus on few-shot in-context learning: models can learn a new task from a few examples during prompting without parameter updates.
- **Datasets**
  - Pretraining datasets:
    - MTB (Medical Textbook Dataset): 4,721 textbooks → 0.8M images + 584M tokens. Interleaved text and images. 95% train / 5% eval.
    - PMC-OA: 1.6M image-caption pairs from PubMed Central Open Access. 1.3M train / 0.16M eval.
  - Evaluation datasets:
    - VQA-RAD: Radiology VQA dataset, custom train/test split to prevent leakage.
    - PathVQA: Pathology VQA dataset.
    - Visual USMLE: 618 complex, multimodal USMLE-style problems (images + vignettes + lab tables). Open-ended rather than multiple-choice.
- **Model and Training**
  - Initialized from OpenFlamingo-9B.
  - ~8.3B total parameters (1.3B trainable, 7B frozen).
  - Multi-GPU training: 8×80GB NVIDIA A100 with DeepSpeed ZeRO Stage 2.
  - Training: 2,700 steps (~6.75 days), batch size 400, gradient accumulation 50.
- **Evaluation & Metrics**
  - Clinical evaluation score: human experts (0–10 scale).
  - BERT similarity score (BERT-sim): automated textual similarity.
  - Exact-match: fraction of generations exactly matching reference (strict, noisy).
  - Baselines:
    - MedVINT: LLaMA-based, visual instruction tuned. Zero-shot & fine-tuned (where dataset allowed).
    - OpenFlamingo: general-domain VLM, zero-shot & few-shot.
- **Results**
  - VQA-RAD: Med-Flamingo few-shot improved clinical score by ~20% over best baseline.
  - PathVQA: All models performed worse (limited pathology pretraining).
  - Visual USMLE: Med-Flamingo few-shot produced most clinically preferred answers. Zero-shot OpenFlamingo performed second best. Exact-match not informative due to long, paragraph-style answers.
  - Overall ranking: Med-Flamingo = 1.67, OpenFlamingo zero-shot = 2.33 (averaged across datasets).
- **Limitations / Observations**

  - Models occasionally hallucinate or generate low-quality responses.
  - Few-shot prompts may leak info from in-context examples.
  - Pathology underrepresented in pretraining datasets.
  - Deduplication required to prevent pretraining-evaluation dataset leakage.
  - Longer Visual USMLE prompts necessitated summarization, sometimes reducing automated metric scores.

- **Key Contributions / Insights**
  - Demonstrates few-shot multimodal generalization for medical VQA.
  - Introduces large-scale curated multimodal datasets for pretraining (MTB, PMC-OA).
  - Develops Visual USMLE benchmark, capturing cross-specialty reasoning and real-world clinical complexity.
  - Shows human evaluation is critical; automated metrics alone may not align with clinical relevance.
- **LitReview Relevance**
  - Best suited for 2.3 Multimodal Fusion Methods (discussion of handling multimodal inputs and interleaving strategies).
  - Also relevant to 3. SOTA Classification / VQA Methods, showing strong few-shot VQA performance on challenging benchmarks.
  - Can contribute to 4. Research Gaps:
    - Need for domain-specific pretraining data, especially in underrepresented fields like pathology.
    - Importance of human evaluation alongside automated metrics in multimodal medical models.

[On Large Visual Language Models for Medical Imaging Analysis: An Empirical Study](https://ieeexplore.ieee.org/document/10614428)

[MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085)

[Few-shot medical image classification with simple shape and texture text descriptors using vision-language models](https://arxiv.org/pdf/2308.04005)

[MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/pdf/2210.10163)

[ViLMedic: aframework for research at the intersection of vision and language in medical AI](https://aclanthology.org/2022.acl-demo.3.pdf)

[CLIP-Lung: Textual Knowledge-Guided Lung Nodule Malignancy Prediction](https://arxiv.org/pdf/2304.08013)

[LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/pdf/2306.00890)

[MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](https://arxiv.org/pdf/2506.00555)

[Advancements in Medical Radiology Through Multimodal Machine Learning: A Comprehensive Overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC12108733/)

[Assessing the performance of zero-shot visual question answering in multimodal large language models for 12-lead ECG image interpretation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11839599/)

[MedFuseNet: An attention-based multimodal deep learning model for visual question answering in the medical domain](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494920/)

[Collaborative Modality Fusion for Mitigating Language Bias in Visual Question Answering](https://pmc.ncbi.nlm.nih.gov/articles/PMC10971294/)

[Histopathology in focus: a review on explainable multi-modal approaches for breast cancer diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11471683/)

[A scoping review on multimodal deep learning in biomedical images and texts](https://pmc.ncbi.nlm.nih.gov/articles/PMC10591890/)

(Will be auto-populated with `pandoc --citeproc` + `references.bib`.)
