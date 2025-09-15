# Multimodal Medical Classification

## Introduction

Multimodal classifiers combine multiple sources, such as images and text, to improve prediction accuracy. Clinical diagnosis often requires integrating imaging studies (e.g., chest X-rays, CT scans) with textual information like radiology reports or clinical notes [@Haq2025_Advancements_MMML_Radiology]. These modalities offer complementary insights: images capture visual manifestations of disease, while text provides clinical context and explicit findings.

Models integrating vision and text outperform uni-modal models [@Sun2023_MultimodalReview]. For instance, textual reports can highlight subtle findings on X-rays, guiding attention, while images can clarify vague text. By fusing both sources, classifiers improve disease prediction. This review surveys methods focusing on fusion strategies (early/late/joint), pretrained vision–language backbones (CLIP, LLaVA, Flamingo), contrastive learning, representative datasets, benchmark results, and ongoing challenges.

## Existing Approaches

### Image-only Classification Methods

Disease detection using only images of the area of concern can yield high performance, but such methods cannot incorporate additional text-based features—such as patient history—that often help doctors make better diagnoses. CheXNet [@Rajpurkar2017_CheXNet] is a deep learning Convolutional Neural Network (CNN) architecture to diagnose pneumonia using chest X-rays. The model is a 121-layer CNN that can output the probability of pneumonia. CheXNet yielded a strong performance in pneumonia classification, outperforming radiologist performance on the same dataset. However, while CNNs can yield strong performance on image classification tasks, they cannot incorporate patient history. CheXNet can highlight areas of an X-ray which indicate the pathology identified, explaining its classification decision. Vision transformers (ViT) can also be used to detect diseases using imaging. A ViT was trained on detecting Alzheimer's disease using brain imaging achieving nearly 96.2% accuracy on the test set [@Lyu2022_ViT_AD]. ViTs can achieve strong performance as they learn using patches of the images to learn relationships between the patches. It outperforms a pretrained CNN model, ResNet18, achieving a 88.5% accuracy on the same test set [@Lyu2022_ViT_AD]. However, ViTs are very deep transformer models and require a significantly large dataset for pretraining. Training on a modest size dataset would lead to overfitting. Like CNNs, ViTs also cannot incorporate text-based features such as the patient's history.

### Text-only Classification Methods

When building multimodal classifiers that combine images with text, it helps to start with a baseline that looks at text alone. Previous research has shown that trained physicians can get around 65% accuracy when making diagnoses using only patient questionnaires, without seeing any medical images [@Uehara2013_DiagnosisQuestionnaire]. More recently, large language models (LLMs) have been tested on similar tasks and were able to reach about 59.1% accuracy when the correct answer was counted as long as it appeared in the top ten predictions [@Tu2025_AMIE].
These results highlight both the value and the limits of text-only methods. For example, the LLM study reports accuracy in terms of “top-10,” which means the correct answer might be buried several guesses down rather than being the first choice [@Tu2025_AMIE]. In real clinical settings, that kind of performance wouldn’t be enough on its own. Similarly, the 65% physician result came from doctors with significant training in general ambulatory care. Less experienced physicians performed much worse, showing that the baseline depends heavily on expertise [@Uehara2013_DiagnosisQuestionnaire].
Altogether, these baselines suggest that while text-only reasoning—whether from humans or algorithms—can provide useful diagnostic clues, it isn’t reliable enough to stand alone. This is where multimodal systems come in. By combining images like radiographs or pathology slides with patient-reported symptoms and notes, we can build models that are not only more accurate but also better at capturing the full picture of a case. Multimodal methods have the potential to go beyond the limitations of a single input type and provide stronger, more trustworthy medical AI tools.

### Multimodal Fusion Methods

#### Fusion Strategies and Model Architectures

Methods for multimodal classification differ in how they merge image and text features [@Sun2023_MultimodalReview]:

- Early fusion (feature-level): concatenates image and text embeddings into a joint representation.
- Late fusion (decision-level) combines outputs of independent image-only and text-only classifiers.
- Joint fusion architectures allow deep cross-modal interaction via shared latent spaces or attention mechanisms.

For example, Gapp et al. fuse chest X-ray and report embeddings using a LLaMA-II backbone, exploring early, late, and mixed fusion pipelines for thoracic disease classification [@Gapp2024_LLaMA2Med]. Vision-language transformers like Med-Flamingo use cross-attention to weigh relevant text for each image region [@Moor2023_MedFlamingo]. LLaVA-Med uses a projection/prefix-style fusion [@Li2023_LLaVA_Med]. CoD-VQA leverages richer modalities to enhance underrepresented ones, reducing modality-specific bias [@Lu2024CoDVQA].

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

## Conclusion

CNNs and ViTs can identify patterns in images that indicate the presence of a pathology. However, other relevant textual features, such as patient history, cannot be incorporated into an image-based classifier [@Rajpurkar2017_CheXNet]. Standalone text-only models also cannot classify diseases with high accuracy, as seen in [@Tu2025_AMIE].

Using both text and images adds valuable context for improved prediction on medical datasets. Multimodal approaches include VLMs that accept image and text tokens in a single sequence [@Moor2023_MedFlamingo], as well as models with dedicated layers for text, image, and their fusion. Our work on predicting answers for images and questions in the OmniMedVQA [@hu2024omnimedvqa] motivates our exploration of using multimodal models to incorporate all available information for prediction.

## References

[//]: <> (Will be auto-populated with `pandoc reports/draft_reports/part1_litreview.md --citeproc --bibliography=references.bib --csl=ieee.csl  -o deliverables/part1/part1_litreview.html`... `pandoc reports/draft_reports/part1_litreview.md -o deliverables/part1/part1_litreview.pdf --pdf-engine=xelatex --citeproc --bibliography=references.bib --csl=ieee.csl -V classoption=twocolumn -V geometry:top=0.75in -V geometry:bottom=0.75in -V geometry:left=0.75in -V geometry:right=0.75in -V fontsize=10pt`)
