# OmniMedVQA Diagnosis Benchmarking Report

## 1. Introduction
- Briefly describe the problem: diagnosis prediction from OmniMedVQA dataset using text-only, image-only, and multimodal models.
- State the dataset splits (train/val/test) and mention if stratified or random sampling was used (with justification).
- Outline the goals: compare unimodal vs multimodal approaches, evaluate performance with appropriate metrics, and assess tradeoffs in resource usage.

---

## 2. Text-Only Baseline
### 2.1 Model Description
- Architecture used (e.g., BERT, DistilBERT, logistic regression on embeddings).
- Input features (questions, candidate answers).
- Training setup (hyperparameters, optimizer, epochs, etc.).

### 2.2 Data Sampling
- If full dataset was not used, describe sampling method (stratified by modality/diagnosis vs. random).
- Report the final sample sizes for train/val/test.

### 2.3 Evaluation
- Metrics: Accuracy, Precision, Recall, F1 (macro/micro if classes are imbalanced).
- Results on **validation** and **test** sets (tables or plots).
- Resource usage (training time, GPU/CPU memory, model size).

### 2.4 Observations
- Strengths (e.g., good on text-driven QAs).
- Weaknesses (e.g., ignores image context, poor generalization on ambiguous questions).

---

## 3. Image-Only Baseline
### 3.1 Model Description
- Architecture used (e.g., ResNet50, EfficientNet).
- Input preprocessing (resize, normalization, augmentations).
- Training setup.

### 3.2 Data Sampling
- Sampling method description (if applied).
- Train/val/test sizes.

### 3.3 Evaluation
- Metrics: Accuracy, Precision, Recall, F1.
- Results on validation and test sets.
- Resource usage.

### 3.4 Observations
- Strengths (e.g., good at visually distinct pathologies).
- Weaknesses (e.g., struggles without textual context).

---

## 4. Multimodal Model (Reproduced from Literature)
### 4.1 Model Description
- Name and citation of the published method (e.g., MDNet, CLIP-based VQA).
- Implementation details (which parts were re-implemented, pretrained components, fusion strategy).
- Training setup.

### 4.2 Data Sampling
- Sampling method description (if applied).
- Train/val/test sizes.

### 4.3 Evaluation
- Metrics: Accuracy, Precision, Recall, F1.
- Results on validation and test sets.
- Resource usage (compute time, memory, model size).

### 4.4 Observations
- Strengths (e.g., combines image + text effectively).
- Weaknesses (e.g., more resource-heavy, harder to interpret).
- Comparison with baselines.

---

## 5. Conclusion
- Summarize performance comparisons between text-only, image-only, and multimodal models.
- Highlight tradeoffs: accuracy vs. resource usage vs. interpretability.
- Discuss implications for diagnosis tasks and possible future work (e.g., lighter multimodal models, interpretability improvements).
