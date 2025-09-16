# Data Exploration of OmniMedVQA

## Introduction

`OmniMedVQA` is a large-scale multimodal medical dataset with 88,996 QA items spanning 42 datasets [@hu2024omnimedvqa]. It combines over 82,000 unique medical images with visual question answering (VQA) items covering five primary question types: Disease Diagnosis, Anatomy Identification, Modality Recognition, Other Biological Attributes, and Lesion Grading. This exploration summarizes dataset statistics, visual examples, modalities, and key challenges for modeling.

## Data Cleaning and Schema Consistency

While inspecting the JSON files, we found that `Chest CT Scan.json` contains a single entry using the key `modality` instead of `modality_type`.

We automatically correct this entry so that `modality` is renamed to `modality_type` for consistency.

## Dataset Statistics

### Number of Samples

- QA Items: 88,996
- Unique images: 82,059
- Datasets represented: 42

### Question Type Distribution

![Question Type Distribution](/reports/figures/question_type_distribution.png)

The dataset is heavily dominated by Disease Diagnosis (55,387 items), followed by Anatomy Identification (16,448) and Modality Recognition (11,565).

Less common types such as Other Biological Attributes (3,498) and Lesion Grading (2,098) may require special attention during modeling to avoid underfitting.

### Ground Truth Answer Distribution

Some question types are heavily skewed toward a few frequent answers:

- Disease Diagnosis: `No` and `No, It's normal.` account for ~7,400 QA items.
- Modality Recognition: `MR` and `CT` dominate.

Some question types also contain answers that appear very rarely (sometimes only once). For example:

- Modality Recognition: "Histopathology." appears 8 times.
- Disease Diagnosis: "Fundus neoplasm" appears once.

This sparsity could make learning on rare classes challenging.

Some semantically identical answers differ in punctuation, capitalization, or minor wording, e.g.:

- `x_ray.` vs `X-ray`
- `Dermoscopic imaging` vs `Dermoscopy` vs `Dermoscopy.`
- `Fundus photography` vs `fundus photography.` vs `fundus photography`

Preprocessing steps such as lowercasing, stripping punctuation, and mapping variants to canonical forms may be beneficial. Despite the long-tail and answer variability, all major modalities and question types are represented, which is promising for building a generalizable multimodal model.

### Dataset-Level Distribution

![Number of QA Items per Dataset](/reports/figures/number_of_qa_items_per_dataset.png)

While RadImageNet alone contributes 56,697 QA items (>60% of the total), several datasets at the bottom (e.g., Pulmonary Chest MC with 38 items) are very small. This imbalance in datasets isn't necessarily an issue as long as all modalities are adequately represented.

### Modalities

![Distribution of Modalities](/reports/figures/distribution_of_modalities_pie_chart.png)

The OmniMedVQA dataset includes 8 distinct modalities. While MR dominates with ~35.8% of QA items, followed by CT (~17.8%) and Ultrasound (~12.3%), the less frequent modalities such as OCT (5.2%), Fundus Photography (6.1%), and Microscopy Images (7.5%) still have a substantial number of QA items (4,646–5,680), which should be sufficient for model training.

Although there is a skew toward MR and CT, all clinically relevant modalities are represented, reducing the risk that models will completely ignore underrepresented modalities. However, care may still be needed to ensure that rare modalities are weighted during training or evaluation.

## Visual Question Answering Examples

![Modailty question from pulmonary chest dataset](/reports/figures/chest-disease-qa.png)

![Disease question from pulmonary chest dataset](/reports/figures/chest-imaging-qa.png)

Within each dataset, there is a diversity in the types of questions being asked for a given image. The first image presents a modality question, which asks about the imaging technique used (such as X-ray, CT, or MR), while the second image is a disease diagnosis question. All VQA items include multiple options relevant to the question posed, the ground truth, and the modality for the image.

![Anatomy question from blood cell datset](/reports/figures/blood-disease-qa.png)

Different modalities for imaging are used across all datasets but the same problem type can appear across different datasets. Questions are framed similarly across datasets creating consistency in the OmniMedVQA dataset. We also see that the quality of imaging can vary between datasets. Some of the images are high quality images using technology such as X-ray or a CT scan. But some of the images are camera pictures of human subjects with some background of the environment. In this example it is a picture taken of the output from a microscope. From the examples seen the text length of questions are short and direct with answer options being just a few words.

## Challenges

![one-ex-answer-distr](/reports/figures/one-ex-answer-distr.png)
![answer-uniformity](/reports/figures/answer-uniformity.png)
Many of the questions show highly skewed answer distributions, with one option dominating while others are rarely chosen. This strong answer bias makes the data harder to interpret and can lead models to overfit to the majority choice rather than learning meaningful patterns.

![images-per-class](/reports/figures/images-per-class.png)
![questions-per-dataset](/reports/figures/questions-per-dataset.png)
Even in the questions and images we can see large numbers of class imbalance. For example, RadImageNet accounts for 55k questions and images with the next closest one being 10k and 3k per plot respectively. Such imbalance makes it harder for models to learn meaningful patterns from underrepresented classes.

## References

[//]: <> (Will be auto-populated with `pandoc reports/draft_reports/part1_dataexploration.md --citeproc --bibliography=references.bib --csl=ieee.csl  -o deliverables/part1/part1_dataexploration.html`... `pandoc reports/draft_reports/part1_dataexploration.md -o deliverables/part1/part1_dataexploration.pdf --resource-path=.:reports:/reports/figures --pdf-engine=xelatex --citeproc --bibliography=references.bib --csl=ieee.csl -V classoption=twocolumn -V geometry:top=0.75in -V geometry:bottom=0.75in -V geometry:left=0.75in -V geometry:right=0.75in -V fontsize=10pt`)
