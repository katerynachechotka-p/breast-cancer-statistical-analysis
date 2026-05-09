# %% [markdown]
# # Statistical Inference: Malignant vs. Benign Cell Concavity
#
# **Research question:** Is the concavity of malignant cells greater than that of benign cells?
#
# **Dataset:** Breast Cancer Wisconsin (Diagnostic) — 569 instances
# Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# %% [markdown]
# ## Appendix A — Data Import

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
from pprint import pprint

df = pd.read_csv("https://course-resources.minerva.edu/uploaded_files/mu/00353110-6347/breast-cancer-wisconsin.csv")
df = df.drop(columns=['id'], inplace=False)
df.head()

# %%
# Check for missing values
if df['concavity_mean'].isnull().values.any():
    print("The 'concavity_mean' column has NaN values.")
else:
    print("The 'concavity_mean' column does not have NaN values.")

# %% [markdown]
# ## Appendix B — Descriptive Statistics & Visualizations

# %%
from utils import my_mean, my_median, my_mode, my_sample_SD, my_range, my_skewness, my_kurtosis

def descriptive_statistics(df):
    malignant_concavity = df[df['diagnosis'] == 'M']['concavity_mean']
    benign_concavity = df[df['diagnosis'] == 'B']['concavity_mean']

    statistics = {
        'Malignant': {
            'Mean': my_mean(malignant_concavity),
            'Median': my_median(malignant_concavity),
            'Mode': my_mode(malignant_concavity),
            'Standard Deviation': my_sample_SD(malignant_concavity),
            'Range': my_range(malignant_concavity),
            'Skewness': my_skewness(malignant_concavity),
            'Excess_Kurtosis': my_kurtosis(malignant_concavity)
        },
        'Benign': {
            'Mean': my_mean(benign_concavity),
            'Median': my_median(benign_concavity),
            'Mode': my_mode(benign_concavity),
            'Standard Deviation': my_sample_SD(benign_concavity),
            'Range': my_range(benign_concavity),
            'Skewness': my_skewness(benign_concavity),
            'Excess_Kurtosis': my_kurtosis(benign_concavity)
        }
    }
    return statistics

statistics = descriptive_statistics(df)
pprint(statistics, sort_dicts=False)

# %%
malignant_concavity = df[df['diagnosis'] == 'M']['concavity_mean']
benign_concavity = df[df['diagnosis'] == 'B']['concavity_mean']

# Separate histograms
plt.hist(malignant_concavity, bins=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
         0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
         edgecolor="black", color="red")
plt.xlabel("Malignant Concavity")
plt.ylim(0, 250)
plt.ylabel("FREQUENCY")
plt.show()

plt.hist(benign_concavity, bins=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
         0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
         edgecolor="black", color="green")
plt.xlabel("Benign Concavity")
plt.ylim(0, 250)
plt.ylabel("FREQUENCY")
plt.show()

# Overlapping histogram
plt.hist(malignant_concavity, bins=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
         0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
         edgecolor="black", color="red", alpha=0.7, label="Malignant")
plt.hist(benign_concavity, bins=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
         0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
         edgecolor="black", color="green", alpha=0.6, label="Benign")
plt.xlabel("Concavity")
plt.ylim(0, 250)
plt.ylabel("FREQUENCY")
plt.legend()
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
data = [benign_concavity, malignant_concavity]
labels = ['Benign', 'Malignant']
plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "firebrick",
                       "markeredgecolor": "firebrick", "markersize": 10},
            medianprops={"linestyle": "--", "color": "darkred", "linewidth": 2},
            boxprops={"facecolor": "lightcoral", "color": "darkred"},
            whiskerprops={"color": "indianred", "linestyle": "-."},
            capprops={"color": "indianred"})
plt.xlabel("Diagnosis", fontsize=12)
plt.ylabel("Concavity Mean", fontsize=12)
plt.show()

# %% [markdown]
# ## Appendix C — T-test Conditions (Normality via Sampling Distribution)

# %%
sample_size = len(malignant_concavity) + len(benign_concavity)

sample_means_malignant = []
sample_means_benign = []

for _ in range(10000):
    malignant_sample = random.choices(malignant_concavity.tolist(), k=len(malignant_concavity))
    benign_sample = random.choices(benign_concavity.tolist(), k=len(benign_concavity))
    sample_means_malignant.append(np.mean(malignant_sample))
    sample_means_benign.append(np.mean(benign_sample))

differences = np.array(sample_means_malignant) - np.array(sample_means_benign)

# Sampling distribution of difference of means
plt.hist(differences, bins=30, alpha=0.5)
plt.xlabel('Difference of Sample Means (Malignant - Benign)')
plt.ylabel('Frequency')
plt.show()

# Normal probability (QQ) plot
res = stats.probplot(differences, plot=plt)
plt.title('')
plt.show()

# %% [markdown]
# ## Appendix D — Significance Testing: Welch's t-test & Cohen's d

# %%
def difference_of_means_test(data1, data2, tails):
    """
    Welch's t-test for difference of means.
    tails: 1 for one-tailed, 2 for two-tailed.
    Returns p-value and Cohen's d effect size.
    """
    n1 = len(data1)
    n2 = len(data2)

    x1 = np.mean(data1)
    x2 = np.mean(data2)

    s1 = np.std(data1, ddof=1)
    s2 = np.std(data2, ddof=1)

    standard_error = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t_score = np.abs((x2 - x1)) / standard_error
    df = min(n1, n2) - 1  # conservative estimate

    p_value = tails * stats.t.cdf(-t_score, df)

    # Cohen's d with pooled SD (unequal group sizes)
    sd_pooled = np.sqrt((s1**2 * (n1 - 1) + s2**2 * (n2 - 1)) / (n1 + n2 - 2))
    cohens_d = (x2 - x1) / sd_pooled

    print('p =', p_value)
    print('d =', cohens_d)

print("Difference-of-means function loaded and ready to use.")
difference_of_means_test(malignant_concavity, benign_concavity, 1)

# %%
# Confidence intervals
n1_m = len(malignant_concavity)
n2_b = len(benign_concavity)
x1_m = np.mean(malignant_concavity)
x2_b = np.mean(benign_concavity)
s1_m = np.std(malignant_concavity, ddof=1)
s2_b = np.std(benign_concavity, ddof=1)

# Malignant CI
SE_m = s1_m / n1_m ** 0.5
t = stats.t.ppf(0.975, n1_m - 1)
print("Threshold t value for 95% confidence =", round(t, 2))
print("Confidence interval of malignant concavity:",
      [round(x1_m - t * SE_m, 2), round(x1_m + t * SE_m, 2)])

# Benign CI
SE_b = s2_b / n2_b ** 0.5
t = stats.t.ppf(0.975, n2_b - 1)
print("Threshold t value for 95% confidence =", round(t, 2))
print("Confidence interval of benign concavity:",
      [round(x2_b - t * SE_b, 2), round(x2_b + t * SE_b, 2)])
