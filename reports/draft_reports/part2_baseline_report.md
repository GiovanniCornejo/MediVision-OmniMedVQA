# OmniMedVQA Diagnosis Benchmarking Report

## 1. Introduction
- Briefly describe the problem: diagnosis prediction from OmniMedVQA dataset using text-only, image-only, and multimodal models.
- State the dataset splits (train/val/test) and mention if stratified or random sampling was used (with justification).
- Outline the goals: compare unimodal vs multimodal approaches, evaluate performance with appropriate metrics, and assess tradeoffs in resource usage.

---

## 2. Text-Only Baseline
### 2.1 Model Description
In our literature review, we identified examples [@Uehara2013_DiagnosisQuestionnaire] [@Tu2025_AMIE] where Large Language Models (LLMs) were trained on patient questionaires to make diagonosis. The issue was the poor performance of 65% accuracy in diagnosis since the model was limited to only textual data of the patient. I used a pretrained BERT model called BiomedBERT which was trained on abstracts from PubMed and full text articles from PubMed Central [@pubmedbert]. We create a model for multiple choice using the pretrained weights from BiomedBERT. For each question, it is provided four options of anwswer choices and it picks the best choice. To train the model, I use the following training arguments:

| Argument | Value | Description |
|-----------|--------|-------------|
| `evaluation_strategy` (`eval_strategy`) | `"epoch"` | Runs evaluation after each training epoch |
| `learning_rate` | `5e-5` | Learning rate for the optimizer |
| `per_device_train_batch_size` | `16` | Number of training examples per device (e.g., per GPU) |
| `per_device_eval_batch_size` | `16` | Number of evaluation examples per device |
| `num_train_epochs` | `2` | Total number of training epochs |
| `weight_decay` | `0.01` | Strength of L2 regularization |

### 2.2 Data Sampling
We filter the dataset to include only examples of questions on the disease diagnosis. As a result we have 55,387 data samples. We split the dataset into test, train and validation sets using a 85%/10%/15% split. The datset is stratified on modality type. We have 42380 samples for training, 7472 for validation, and 5535 for test. 

### 2.3 Evaluation
The reported results from our training shows that the model does no generalize, achieving similar losses for both training and validation. It also achives a very high accuracy of 0.995 in the validation set. 
| Epoch | Validation Accuracy | Validation Loss | Training Loss|
|-------|---------------------|-----------------|--------------|
|1|0.993041|0.030241|0.036300|
|2|0.994781|0.021077|0.022700|

Evaluating the trained model on the test set we get the following results:
| Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro)|
|-------|---------------------|-----------------|--------------|
|0.9906|0.9909| 0.9913|0.9911|

Since there are four possible options or classes the model decides on, we compute a macro average for precision, recall, and f1 scores. 

### 2.4 Observations
The model performs very well on the text based questions. This highlights the simplicity of the datset's questions. Examining the questions that appear in the test set and the training dataset, many questions are very similar with similar answer options. Without any images and only the text of the questions and the options, the model performs very well making correct guesses based on similar questions that appeared in the training data. The number of unique question strings that appear in the test datset is 2316. Of those unique questions in the test set, 1868 questions appear in the training dataset. Computing the accuracy only on questions that appeared in training dataset and those that did not we get:
|Test Set Subset| Accuracy|
|----------------------------|---------|
|Questions In Train|0.992|
|Questions Not In Train|0.979|

The model does perform slightly worse on unseen questions. But it still performs very well as these different questions may have similar answer options to other questions or the question is similar to another question in the dataset.

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
