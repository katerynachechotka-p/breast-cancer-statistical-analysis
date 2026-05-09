# breast-cancer-statistical-analysis
Statistical analysis of the Breast Cancer Wisconsin (Diagnostic) Dataset using Python. Two independent analyses are included:

Regression Analysis — OLS linear regression predicting cell concavity from compactness
Statistical Inference — Difference of means testing comparing malignant vs. benign cell concavity


Repository Structure
breast-cancer-statistical-analysis/
│
├── regression_analysis.ipynb       # OLS regression, Pearson's r, residual diagnostics
├── statistical_inference.ipynb     # t-test, Cohen's d, confidence intervals
├── utils.py                        # Shared statistical functions implemented from scratch
└── README.md

Analysis 1: Regression Analysis
Research question: Can a linear model predict breast cell concavity based on compactness?
Methods:

Pearson's r correlation
Ordinary Least Squares (OLS) regression via statsmodels
Residual diagnostics: QQ plot, histogram of residuals, residuals vs. fitted values
Significance testing: t-score, p-value, 95% confidence interval for slope

Key results:

Pearson's r = 0.883 (strong positive linear relationship)
R² = 0.78 (78% of variance in concavity explained by compactness)
Slope = 1.333, 95% CI: [1.275, 1.391], p-value ≈ 0
Heteroscedasticity identified — OLS assumptions partially violated
Suggested improvements: log transformation, weighted least squares


Analysis 2: Statistical Inference
Research question: Is the concavity of malignant cells greater than that of benign cells?
Methods:

Descriptive statistics by diagnosis group
Welch's t-test (unequal variance)
Cohen's d effect size
95% confidence intervals for each group mean
Sampling distribution visualization (bootstrap, n=10,000)

Key results:

Malignant mean concavity: 0.161 vs. Benign: 0.046
T-score = 20.3, p-value = 6.56e-52 (reject null hypothesis)
Cohen's d = 2.003 (large effect size)
95% CI Malignant: [0.15, 0.17], Benign: [0.04, 0.05] — non-overlapping


Tools & Libraries

Python 3, Jupyter Notebook
pandas, NumPy, SciPy, statsmodels
Matplotlib, Seaborn


Dataset
Breast Cancer Wisconsin (Diagnostic) Data Set — 569 instances, 30 features derived from fine needle aspirate (FNA) images.
Source: Kaggle / UCI ML Repository
