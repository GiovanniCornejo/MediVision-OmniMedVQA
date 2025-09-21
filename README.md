# MediVision-OmniMedVQA

## 📌 Project Overview

**MediVision-OmniMedVQA** is a research project exploring **multimodal medical classification** using the [OmniMedVQA benchmark](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA).

The goal is to design, benchmark, and evaluate models that combine **medical imaging** (vision) with **textual clinical context** (language) to improve diagnostic accuracy, interpretability, and robustness.

Our goal is to:

- Implement **image-only** and **text-only** baselines.
- Reproduce an existing multimodal method from the literature.
- Develop and evaluate our own multimodal fusion approach.
- Compare approaches across **classification** tasks.

### 🩺 Dataset

The _full_ OmniMedVQA contains ~118,000 medical images and ~128,000 associated question-answer pairs spanning 73 datasets and 12 different medical imaging modalities. However, only the open-access portion is included here — restricted-access datasets require applying for access directly from their original sources.

The open-access subset of OmniMedVQA includes ~82,000 medical images and ~89,000 question–answer pairs, covering 42 publicly available datasets. This subset provides broad coverage across diverse imaging modalities and clinical question types while remaining accessible without special permissions.

- **Modalities**: MR (Mag-netic Resonance Imaging), CT (Computed Tomography), Ultrasound, X-Ray, Dermoscopy, Microscopy Images, Fundus Photography, OCT (Optical Coherence Tomography)
- **Question Types**: Disease Diagnosis, Anatomy Identification, Modality Recognition, Other Biological Attributes, Lesion Grading

> **Note**: You will need to manually download the dataset from the official source.

## ⚙️ Setup & Installation

We recommend creating a dedicated Python environment to ensure reproducibility. Below are instructions using **conda** (recommended) and **venv**.

1. Clone the repository

```bash
git clone https://github.com/GiovanniCornejo/MediVision-OmniMedVQA.git
cd MediVision-OmniMedVQA
```

2. Create a new environment

**Using Conda (recommended)**:

```bash
conda create -n medivision python=3.10 -y
conda activate medivision
```

**Using venv**:

```bash
python3 -m venv medivision
source medivision/bin/activate  # Linux/Mac
medivision\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. (Optional) Jupyter Notebook kernel

If you plan to run the notebooks interactively:

```bash
pip install ipykernel
python -m ipykernel install --user --name=medivision --display-name "Python (medivision)"
```

### 📂 Data Directory Setup

By default, the project expects datasets to be stored under `MediVision-OmniMedVQA/data/`. This path is relative to the repo root, so make sure your **current working directory** is the repo front when running Jupyter notebooks or Python scripts:

```bash
cd Medivision-OmniMedVQA
jupyter notebook .
```

If you want to keep the dataset elsewhere, you can override the default path by setting the environment variable `OMNIMEDVQA_DATA_DIR`. **Important**: the variable must be set in the same terminal session from which you start Jupyter Notebook or run your Python scripts, so the process can see it. For example:

```bash
cd Medivision-OmniMedVQA

export OMNIMEDVQA_DATA_DIR=~/datasets/OmniMedVQA  # Linux/Mac
$env:OMNIMEDVQA_DATA_DIR="D:\Datasets\OmniMedVQA" # Windows

jupyter notebook .
```

### 🧰 Dependencies

This project relies on the following Python packages:

- **Core / General Purpose**:
  - Python >= 3.10
  - numpy
  - pandas
- **Data Visualization**:
  - matplotlib
  - seaborn
- **Dataset Handling**:
  - datasets (Hugging Face)
  - Pillow
- **Machine Learning**
  - scikit-learn
  - torch (PyTorch)
  - torchvision
  - transformers
- **Utilities / Misc**:
  - tqdm
- **Notebook / Interactive Environment**:
  - jupyter
  - ipykernel
  - nbstripout

> **Note:** If you want GPU support for faster training, install a CUDA-enabled version of PyTorch compatible with your GPU and drivers. See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for instructions.

### 🚀 Quick Start

After setting up the environment and downloading the dataset:

1. Run the data exploration notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. Run a baseline experiment (image-only):

```bash
python src/train.py --config experiments/configs/baseline_image.json
```

3. Evaluate the results:

```bash
python src/evaluate.py --config experiments/configs/baseline_image.json
```

## 🏗️ Repository Structure

```
MediVision-OmniMedVQA/
│
├── README.md                 <- Project overview, setup instructions
├── references.bib            <- BibTeX file with all references and notes
├── requirements.txt          <- Python dependencies
├── .gitignore
├── .gitattributes
│
├── notebooks/                <- Jupyter notebooks for different project stages
│   ├── 01_part1_modality_analysis.ipynb
│   ├── 01_part2_visualizing_examples.ipynb
│   ├── 01_part3_dataset_challenges.ipynb
│   ├── 02_part1_baselines.ipynb
│   ├── 02_part2_reproduce_multimodal.ipynb
│   ├── 03_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── src/                      <- Modular Python code
│   ├── config.py             <- Project-wide constants
│   ├── data.py               <- Dataset loading / preprocessing
│   ├── models.py             <- Baseline + multimodal model architectures
│   ├── train.py              <- Training loop functions
│   └── evaluate.py           <- Evaluation metrics & utilities
│
├── data/                     <- (Gitignored) placeholder for datasets
│
├──experiments/               <- Configs & logs
│   ├── configs/              <- JSON/YAML configs for experiments
│   └── logs/                 <- Training logs, saved metrics
│
├── reports/                  <- Outputs and deliverables
│   ├── figures/              <- Visualizations
│   ├── tables/               <- Metrics/results
│   └── draft_reports/        <- IEEE-style report drafts
│
└── deliverables/             <- Finalized pieces to submit
    ├── part1/
    │   ├── part1_exploration.pdf
    │   └── part1_litreview.pdf
    ├── part2/
    │   └── part2_benchmarking.pdf
    ├── part3/
    │   └── part3_method_description.pdf
    └── final_report/
        └── final_report.pdf
```

## 🔜 Project Status

### 🔬 1. Data Exploration & Literature Review

- [x] Load and explore OmniMedVQA dataset using Hugging Face
  - [x] Inspect folder structure and image formats
  - [x] Count number of images and questions per modality
- [x] Analyze class distributions and modalities
- [x] Visualize image-text pairs / QA items
- [x] Identify dataset challenges (class imbalance, ambiguous prompts, domain differences)
- [x] Write Data Exploration report (max 2 pages)
- [x] Conduct literature review of multimodal medical classification
- [x] Write Literature Review summary (max 2 pages)

### 📊 2. Benchmarking

- [ ] Implement image-only baseline
  - [ ] Preprocess images (resize, normalize, augment)
  - [ ] Train and evaluate baseline model
- [ ] Implement text-only baseline
  - [ ] Preprocess text
  - [ ] Train and evaluate baseline model
- [ ] Reproduce one published multimodal method
- [ ] Evaluate models on validation & test splits (accuracy, precision, recall, F1-score)
- [ ] Compare models across modalities and metrics
- [ ] Write Benchmarking report (max 3 pages)

### ⚡ 3. Development of Own Method

- [ ] Define goal and propose multimodal architecture
- [ ] Implement own model using PyTorch / Hugging Face transformers
- [ ] Ensure reproducibility (modular code, fixed seeds, documented pipelines)
- [ ] Write Method Description document (max 3 pages)

### 📝 4. Comparative Evaluation & Report Writing

- [ ] Evaluate all models on same test splits
- [ ] Generate confusion matrices and consistent metrics
- [ ] Conduct ablation studies for modality / component contributions
- [ ] Analyze limitations and propose future improvements
- [ ] Write full IEEE-style report (8–12 pages including figures & references)
  - [ ] Compile References in IEEE format
  - [ ] Draft Abstract (~200 words)
  - [ ] Write Introduction (context, problem statement, contributions)
  - [ ] Include Related Work (summary of literature review)
  - [ ] Summarize Dataset & Exploratory Analysis (Part #1 outputs)
  - [ ] Describe Baseline Methods (Part #2 outputs)
  - [ ] Detail Proposed Method (Part #3 outputs)
  - [ ] Document Experiments (evaluation setup, metrics, results)
  - [ ] Write Discussion (interpretation, strengths, limitations)
  - [ ] Write Conclusion & Future Work
  - [ ] Review formatting and finalize PDF
- [ ] Package code & supplementary materials for submission

## 📚 Citations / References

This project builds upon several datasets, papers, and methods in the multimodal medical AI domain.  
For a complete list of references, including BibTeX entries, see [references.bib](references.bib).

Key references include:

- **OmniMedVQA, 2024** – Main dataset for this project; includes multiple modalities and question types.
- **Gapp et al., 2024** – Multimodal disease classification using OpenI X-rays + reports; informs baseline and fusion architecture.
- **Med-Flamingo, 2023** – Few-shot vision-language learner for medical VQA; inspiration for multimodal training strategies.

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- Giovanni Cornejo
