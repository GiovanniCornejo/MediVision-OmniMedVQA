# Multimodal Medical Classification

## Introduction

Multimodal classifiers combine multiple sources, such as images and text, to improve prediction accuracy. Clinical diagnosis often requires integrating imaging studies (e.g., chest X-rays, CT scans) with textual information like radiology reports or clinical notes [@Haq2025_Advancements_MMML_Radiology]. These modalities offer complementary insights: images capture visual manifestations of disease, while text provides clinical context and explicit findings.

Models integrating vision and text outperform uni-modal models [@Sun2023_MultimodalReview]. For instance, textual reports can highlight subtle findings on X-rays, guiding attention, while images can clarify vague text. By fusing both sources, classifiers improve disease prediction. This review surveys methods focusing on fusion strategies (early/late/joint), pretrained vision–language backbones (CLIP, LLaVA, Flamingo), contrastive learning, representative datasets, benchmark results, and ongoing challenges.

## Existing Approaches

### Image-only Classification Methods (TODO)

- CNN/ViT-based models applied to medical imaging classification tasks.
- Strengths: strong visual pattern recognition.
- Limitations: cannot incorporate textual or clinical context, limiting performance on complex cases.

### Text-only Classification Methods (TODO)

- Large language models (LLMs) used for classification from clinical notes or prompts.
- Strengths: captures linguistic patterns.
- Limitations: ignores imaging evidence, may misclassify visually subtle pathologies.

### Multimodal Fusion Methods

#### Fusion Strategies and Model Architectures

Methods for multimodal classification differ in how they merge image and text features [@Sun2023_MultimodalReview]:

- Early fusion (feature-level): concatenates image and text embeddings into a joint representation.
- Late fusion (decision-level) combines outputs of independent image-only and text-only classifiers.
- Joint fusion architectures allow deep cross-modal interaction via shared latent spaces or attention mechanisms.

For example, Gapp et al. fuse chest X-ray and report embeddings using a LLaMA-II backbone, exploring early, late, and mixed fusion pipelines for thoracic disease classification [@Gapp2024_LLaMA2Med]. Vision-language transformers like Med-Flamingo use cross-attention to weigh relevant text for each image region [@Moor2023_MedFlamingo] LLaVA-Med [@Li2023_LLaVA_Med] uses a projection/prefix-style fusion. CoD-VQA leverages richer modalities to enhance underrepresented ones, reducing modality-specific bias [@Lu2024CoDVQA].

Large VLMs such as BiomedCLIP and OpenFlamingo achieve strong zero-/few-shot performance without fine-tuning [@Van2024_LargeVLMsMed], and Med-Flamingo adapts quickly to new radiology tasks [@Moor2023_MedFlamingo]. These strategies use complementary data to boost accuracy, although they require carefully aligned training data and can be sensitive to modality-specific noise.

#### Contrastive Learning

Contrastive learning approaches align image and text embeddings. Wang et al. demonstrate improved downstream classification with contrastive training on chest X-rays and reports [@Wang2022_MedCLIP]. Lei et al. introduce CLIP-Lung uses radiology prompts and disease-specific attributes for lung nodule malignancy prediction, achieving SOTA performance on LIDC-IDRI [@Lei2023_CLIP_Lung]. These methods also provide a strong initialization for zero- or few-shot adaptation and help models learn meaningful cross-modal embeddings even when labeled examples are scarce.

#### Datasets

Key multimodal datasets include:

- Chest X-ray: OpenI and MIMIC-CXR contain thousands of frontal X-rays with paired radiology reports and labels for common findings [@Gapp2024_LLaMA2Med].
- Lung CT nodules: LIDC-IDRI provides lung CT scans with annotated nodules. Recent work augments LIDC with textual nodule descriptions to classify nodules as benign or malignant [@Lei2023_CLIP_Lung].
- Ultrasound: UDIAT (breast ultrasound) has images with diagnostic labels. Byra et al. show that even simple text descriptors combined with CLIP enable few-shot ultrasound classification [@Byra2023_FewShotVLM].
- Histopathology: Emerging datasets pair pathology images with clinical summaries; for example, multimodal breast cancer challenges include histology slides accompanied by relevant text [@Abdullakutty2024_MultiModalHistopathology].

These examples cover typical classification tasks (single-label and multi-label) in radiology and related fields. Most current models are evaluated on such radiology benchmarks, but diversity across modalities (e.g. MRI, pathology) and tasks remains limited, highlighting the need for broader multimodal datasets.

## State-of-the-Art Models

Recent multimodal classifiers achieve strong results on benchmark tasks:

- LLaMA-II fusion [@Gapp2024_LLaMA2Med]: AUC 0.971 on OpenI chest X-ray classification using fused image-report inputs, surpassing unimodal baselines.
- CLIP-Lung [@Lei2023_CLIP_Lung]: SOTA on LIDC-IDRI nodule classification with textual knowledge guidance.
- MedCLIP [@Wang2022_MedCLIP]: Outperforms previous self-supervised methods in image–text retrieval and classification.
- Large VLMs: BiomedCLIP and OpenFlamingo achieve competitive zero-/few-shot performance compared to CNNs without the need for fine-tuning [@Van2024_LargeVLMsMed], while Med-Flamingo adapts quickly to new radiology tasks [@Moor2023_MedFlamingo].

In general, vision–language models (both fine-tuned and zero-shot) consistently outperform uni-modal baselines on these benchmarks [@Wang2022_MedCLIP], demonstrating the benefit of multimodal integration.

## Challenges and Research Gaps

Key challenges remain in multimodal medical classification:

- Data scarcity, modality, and domain shift: High-quality paired image–text examples are limited and often skewed toward certain modalities (e.g., more X-rays than matched reports), which can bias training. Models trained on a single dataset or domain may also fail to generalize to other modalities (e.g., X-ray to CT) [@Sun2023_MultimodalReview]. Ensuring robust performance across imaging types, domains, and patient populations remains an open challenge.
- Interpretability and trust: Modern multimodal models (especially large transformers) are largely opaque. Works like attention maps in MedFuseNet offers partial insight [@Sharma2021_MedFuseNet], but systematically explaining how image and text combine to yield a prediction remains difficult. This lack of transparency hinders clinical acceptance.
- Bias and hallucination: VLMs may over-rely on textual priors or external knowledge, sometimes ignoring the image and producing "hallucinated" findings [@Xia2025_MMedRAG; @Moor2023_MedFlamingo], e.g., zero-shot ECG VQA models overpredict normal rhythms [@Seki2025_ZeroShotECG]. Developing methods to ensure predictions are truly grounded in data is crucial.
- Resource and annotation costs: Training and fine-tuning large multimodal models require significant compute, and curating labeled image–text datasets in healthcare is labor-intensive. These practical constraints slow the development and evaluation of new methods.

Addressing these gaps is an active research area. For example, retrieval-augmented models like MMed-RAG incorporate external knowledge to reduce hallucination [@Xia2025_MMedRAG]. Expanding multimodal datasets and developing techniques to interpret multimodal reasoning will be key to deploying reliable clinical classifiers in practice.

## Conclusion (TODO)

- Image-only and text-only models are limited for multimodal classification tasks.
- Fusion-based multimodal methods outperform unimodal approaches, but generalization and interpretability challenges remain.
- Motivates your group’s work: developing and benchmarking novel multimodal classifiers for FLARE25, with a focus on both single-label and multi-label classification.

## References

[//]: <> (Will be auto-populated with `pandoc reports/draft_reports/part1_litreview.md --citeproc --bibliography=references.bib --csl=ieee.csl  -o deliverables/part1/part1_litreview.html`...)
