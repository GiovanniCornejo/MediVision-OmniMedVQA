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
We reproduced the multimodal VQA model introduced in [@Moor2023_Flare] (*Foundation Models for Generalist Medical AI*), which adapts the **OpenFlamingo** architecture to combine medical images with natural language questions and answers.  

- **Name and Citation**: OpenFlamingo-based multimodal VQA [Moor et al. 2023].  
- **Original vs. Our Implementation**:  
  - The original paper used **OpenFlamingo-9B**, which integrates a ViT-G/14 vision encoder with a large language model.  
  - Due to time and storage constraints, we used the lighter **OpenFlamingo-3B** variant (`openflamingo/OpenFlamingo-3B-vitl-mpt1b`). This model follows the same architectural design but is significantly smaller and more manageable for our setup.  
- **Model Components**:  
  - **Vision Encoder**: Pretrained CLIP ViT-L/14.  
  - **Language Encoder**: Pretrained MPT-1B-RedPajama-200B.  
  - **Fusion Strategy**: Cross-attention layers inserted into the transformer stack (matching OpenFlamingo).  
  - **Parameter-Efficient Tuning**: LoRA adapters added to the language model for efficient fine-tuning.  
- **Training Setup**:  
  - Optimizer: AdamW (`lr=1e-5`).  
  - Loss: CrossEntropy with padding mask.  
  - Epochs: **3** (shortened from the original setup to save compute).  
  - Batch sizes: 5 (train) / 2 (eval).  
  - Runtime: ~**170 minutes** for the reduced dataset run.  

In summary, we scaled down the model size, number of epochs, and dataset size for feasibility, but preserved the **core architecture** (vision encoder, language encoder, fusion layers, and LoRA fine-tuning) to remain faithful to the original design.  

---

### 4.2 Data Sampling
We adapted the dataset preparation strategy from the paper, with modifications for efficiency:  

- **Dataset**: OmniMedVQA.  
- **Splits**:  
  - **90%** of data reserved for training/validation, **10%** for testing.  
  - Within the training portion, an **85/15 split** was applied for train/validation.  
- **Subsampling**: To save time and storage, we randomly sampled a subset of the dataset. This enabled training to complete within ~170 minutes on a single CPU while still allowing meaningful evaluation.  

---

### 4.3 Evaluation

Perform evaluation on both validation and held-out test splits.  (Still need)

- **Metrics**: Accuracy, Precision, Recall, and F1.  

| Accuracy | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 0.80     | 0.83      | 0.86   | 0.84 |

---

### 4.4 Observations
- **Comparison with Text Baseline**:  
  - The **text-only baseline** achieved higher direct accuracy than our multimodal model under the current constraints.  
  - This is expected because many questions in the dataset can be answered from textual information alone, and the multimodal model was limited by **smaller model size (3B vs. 9B)**, **dataset subsampling**, and only **3 training epochs**.  
  - Despite these limitations, the multimodal model still reached **80% accuracy**, which is strong given the reduced setup. With more data, longer training runs, and the full 9B model, we expect the multimodal model to surpass the text baseline, as it can leverage both visual and textual reasoning.  

- **Strengths**:  
  - Successfully reproduced the OpenFlamingo design (vision encoder + language encoder + fusion).  
  - LoRA adapters enabled efficient fine-tuning on limited resources.  
  - Demonstrated competitive results even with reduced scale.  

- **Weaknesses / Practical Challenges**:  
  - Underperformed relative to the text-only baseline due to reduced scale and short training.  
  - High compute cost (170 minutes for 3 epochs) even in reduced form.  
  - **Environment fragility**: Running OpenFlamingo required **exact Python and module versions** (PyTorch, Hugging Face Transformers, and dependencies).  
    - Incorrect versions prevented the model from downloading or executing.  
    - This version sensitivity made reproducibility more difficult than with the lighter baselines.  


---

## 5. Conclusion
- Summarize performance comparisons between text-only, image-only, and multimodal models.
- Highlight tradeoffs: accuracy vs. resource usage vs. interpretability.
- Discuss implications for diagnosis tasks and possible future work (e.g., lighter multimodal models, interpretability improvements).
