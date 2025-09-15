# Data Exploration of OmniMedVQA

## Introduction (TODO)
We are using the dataset OmniMedVQA which contains over 100,000+ images and question answer (qa) items from 73 different medical datasets [@hu2024omnimedvqa]. Examining the dataset finds that the dataset contains [Todo: number of classes for question] classes of questions over all datasets. 

## Dataset Statistics (TODO)
- The types of problem types and number of classes
- The types of modalites and number of
- The number of images in each datset
- 

## Visual Question Answering Examples

The two examples of visual question answering below are from the pulmonary Chest MC dataset.  
![chest-disease-qa](/assets/chest-disease-qa.png){ width=500px }
![chest-imaging-qa](/assets/chest-imaging-qa.png){ width=500px }

Within each dataset, there is a diversity in the types of questions being asked for a given image. The first image presents a modality question, which asks about the imaging technique used (such as X-ray, CT, or MRI), while the second image is a disease diagnosis question. All VQA items include multiple options relevant to the question posed, the ground truth, and the modality for the image. 

An example from the dataset covid 19 is shown below:
![covid-imaging-qa](/assets/covid-imaging-qa.png){ width=500px }

Different modalities for imaging are used across all datasets but the same problem type can appear across different datasets. Questions are framed similarily across datasets creating consistency in the OmniMedVQA dataset.  

## Challenges

![one-ex-answer-distr](/assets/one-ex-answer-distr.png){ width=500px }
![answer-uniformity](/assets/answer-uniformity.png){ width=500px }
Many of the questions show highly skewed answer distributions, with one option dominating while others are rarely chosen. This strong answer bias makes the data harder to interpret and can lead models to overfit to the majority choice rather than learning meaningful patterns.

![images-per-class](/assets/images-per-class.png){ width=500px }
![questions-per-dataset](/assets/questions-per-dataset.png){ width=500px }
Even in the questions and images we can see large numbers of class imbalance. For example, RadImageNet accounts for 55k questions and images with the next closest one being 10k and 3k per plot respectively. Such imbalance makes it harder for models to learn meaningful patterns from underrepresented classes.

## References

[//]: <> (Will be auto-populated with `pandoc reports/draft_reports/part1_dataexploration.md --citeproc --bibliography=references.bib --csl=ieee.csl  -o deliverables/part1/part1_dataexploration.html`...)
