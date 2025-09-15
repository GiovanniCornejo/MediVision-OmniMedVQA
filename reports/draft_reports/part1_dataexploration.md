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

![Question Type Distribution](/assets/question_type_distribution.png)

The dataset is heavily dominated by Disease Diagnosis (55,387 items), followed by Anatomy Identification (16,448) and Modality Recognition (11,565).

Less common types such as Other Biological Attributes (3,498) and Lesion Grading (2,098) may require special attention during modeling to avoid underfitting.

### Ground Truth Answer Distribution

- Modality Recognition: Dominated by MRI (3,541) and CT (1,817); rare answers like Histopathology. (8) appear only a few times.
- Disease Diagnosis: Skewed towards "No" (3,784) and "No, It's normal." (3,664), with rare diagnoses appearing only once.
- Other question types: Similar long-tail distributions exist, highlighting potential challenges in modeling rare answers.

### Dataset-Level Distribution

- Top datasets: RadImageNet (56,697 QA items), Retinal OCT-C8 (4,016)
- Least represented datasets: Pulmonary Chest MC (38 QA items), NLM Malaria (75)

Dataset imbalance may affect model performance on small datasets but all modalities are represented.

### Modalities

MR and CT dominate, but all major clinical modalities are represented.

## Visual Question Answering Examples

The two examples of visual question answering below are from the pulmonary Chest MC dataset.  
![chest-disease-qa](/assets/chest-disease-qa.png){ width=500px }
![chest-imaging-qa](/assets/chest-imaging-qa.png){ width=500px }

Within each dataset, there is a diversity in the types of questions being asked for a given image. The first image presents a modality question, which asks about the imaging technique used (such as X-ray, CT, or MRI), while the second image is a disease diagnosis question. All VQA items include multiple options relevant to the question posed, the ground truth, and the modality for the image.

This example is from a covid imaging dataset.
![covid-imaging-qa](/assets/covid-imaging-qa.png){ width=500px }

Different modalities for imaging are used across all datasets but the same problem type can appear across different datasets. Questions are framed similarily across datasets creating consistency in the OmniMedVQA dataset. We also see that the quality of imaging can vary between datasets. Some of the images are high quality images using technology such as X-ray or a CT scan. But some of the images are camera pictures of human subjects with some background of the environment. In this example it is a picture taken of the output from a microscope. From the examples seen the text length of questions are short and direct with answer options being just a few words.

## Challenges

![one-ex-answer-distr](/assets/one-ex-answer-distr.png){ width=500px }
![answer-uniformity](/assets/answer-uniformity.png){ width=500px }
Many of the questions show highly skewed answer distributions, with one option dominating while others are rarely chosen. This strong answer bias makes the data harder to interpret and can lead models to overfit to the majority choice rather than learning meaningful patterns.

![images-per-class](/assets/images-per-class.png){ width=500px }
![questions-per-dataset](/assets/questions-per-dataset.png){ width=500px }
Even in the questions and images we can see large numbers of class imbalance. For example, RadImageNet accounts for 55k questions and images with the next closest one being 10k and 3k per plot respectively. Such imbalance makes it harder for models to learn meaningful patterns from underrepresented classes.

## References

[//]: <> (Will be auto-populated with `pandoc reports/draft_reports/part1_dataexploration.md --citeproc --bibliography=references.bib --csl=ieee.csl  -o deliverables/part1/part1_dataexploration.html`...)
