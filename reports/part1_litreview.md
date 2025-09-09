# Literature Review: Multimodal Medical Classification

## 1. Introduction

Briefly introduce the scope of the literature review:

- Why multimodal medical classification is important.
- How vision-language models (VLMs) are emerging in this field.
- What this summary will cover (methods, SOTA baselines, research gaps).

---

## 2. Existing Approaches

### 2.1 Image-only Methods

- Examples of CNN/ViT-based models applied to medical imaging.
- Their strengths (e.g., visual pattern recognition) and limitations (lack of context from text).

### 2.2 Text-only Methods

- Use of large language models (LLMs) for diagnostic text/QA tasks.
- Pros: linguistic understanding. Cons: ignores imaging evidence.

### 2.3 Early & Late Fusion Methods

- Describe multimodal fusion strategies.
- Cite works like Gapp et al. (2024), Med-Flamingo (2023).
- Note performance gains and where they excel/fail.

---

## 3. State-of-the-Art Baselines

- Summarize reported benchmarks on multimodal datasets (e.g., OpenI, FLARE25, MMMED).
- Include key metrics (AUC, accuracy, etc.).
- Highlight best-performing VLMs or multimodal fusion frameworks.
- Emphasize reproducibility (open datasets/models vs. proprietary).

---

## 4. Research Gaps

- Data limitations (imbalances, ambiguous prompts).
- Model generalizability across modalities/domains.
- Interpretability and trustworthiness in clinical settings.
- Computational/resource challenges.

---

## 5. Conclusion

- Restate main findings: image-only/text-only are limited; multimodal approaches outperform but have open challenges.
- Motivate group’s work: exploring novel multimodal architectures and benchmarking against SOTA.

---

## References

(Will be auto-populated with `pandoc --citeproc` + `references.bib`.)
