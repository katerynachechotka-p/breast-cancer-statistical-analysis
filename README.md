# Breast Cancer Wisconsin Dataset — Statistical Analysis in Python

Statistical analysis of the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) using Python. Two independent analyses are included, each implemented from first principles with custom statistical functions.

---

## Repository Structure

```
breast-cancer-statistical-analysis/
│
├── regression_analysis.py           # OLS regression, Pearson's r, residual diagnostics
├── statistical_inference.py         # Welch's t-test, Cohen's d, confidence intervals
├── utils.py                         # Custom statistical functions (mean, SD, skewness, etc.)
├── regression_analysis_report.pdf   # Full written report: Analysis 1
├── statistical_inference_report.pdf # Full written report: Analysis 2
└── README.md
```

---

## Analysis 1: Regression Analysis

**Research question:** Can a linear model predict breast cell concavity based on compactness?

### Methods
- Pearson's r correlation
- Ordinary Least Squares (OLS) regression via `statsmodels`
- Full residual diagnostics: QQ plot, histogram of residuals, residuals vs. fitted values
- Significance testing: t-score, p-value, 95% confidence interval for slope

### Key Results

| Metric | Value |
|--------|-------|
| Pearson's r | 0.883 |
| R² | 0.780 |
| Slope (β₁) | 1.333 |
| 95% CI for slope | [1.275, 1.391] |
| p-value | ≈ 0 |

- Strong positive linear relationship between compactness and concavity
- Heteroscedasticity identified for compactness > 0.15 — OLS assumptions partially violated
- Suggested improvements: log transformation, weighted least squares

📄 [Full report](regression_analysis_report.pdf)

---

## Analysis 2: Statistical Inference

**Research question:** Is the mean concavity of malignant cells greater than that of benign cells?

### Methods
- Descriptive statistics by diagnosis group (mean, SD, skewness, excess kurtosis)
- Welch's t-test (unequal variance between groups)
- Cohen's d effect size
- 95% confidence intervals for each group mean
- Sampling distribution visualization (bootstrap, n = 10,000)

### Key Results

| Metric | Malignant | Benign |
|--------|-----------|--------|
| Mean concavity | 0.161 | 0.046 |
| Standard deviation | 0.075 | 0.043 |
| 95% CI | [0.15, 0.17] | [0.04, 0.05] |

| Test | Result |
|------|--------|
| T-score | 20.3 |
| p-value | 6.56 × 10⁻⁵² |
| Cohen's d | 2.003 (large effect) |

- Null hypothesis rejected — malignant cells have significantly greater concavity
- Non-overlapping confidence intervals confirm practical significance
- Large Cohen's d (2.003) indicates strong real-world distinction between groups

📄 [Full report](statistical_inference_report.pdf)

---

## Technical Implementation

All statistical functions (mean, median, mode, standard deviation, skewness, excess kurtosis) are implemented from scratch in [`utils.py`](utils.py) using algebraic formulas, without relying on library shortcuts. Results are cross-validated against `scipy` and `statsmodels` outputs.

### Tools & Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `NumPy` | Numerical computation |
| `SciPy` | Statistical tests and distributions |
| `statsmodels` | OLS regression modeling |
| `Matplotlib` / `Seaborn` | Data visualization |

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Data Set**
- 569 instances, 30 features
- Features derived from digitized fine needle aspirate (FNA) images
- Source: [UCI ML Repository via Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## About

This work was completed as part of independent quantitative research during undergraduate studies in Physics and Computer Science at Minerva University. Both analyses are fully reproducible — all code is documented and results are validated against established statistical software outputs.
