# %% [markdown]
# # Regression Analysis: Predicting Breast Cell Concavity from Compactness
#
# **Research question:** Can a linear model predict breast cell concavity based on compactness?
#
# **Dataset:** Breast Cancer Wisconsin (Diagnostic) — 569 instances
# Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# %% [markdown]
# ## Appendix A — Data Loading & Preprocessing

# %%
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("https://course-resources.minerva.edu/uploaded_files/mu/00353110-6347/breast-cancer-wisconsin.csv")
df = df.drop(columns=["id"], inplace=False)
df.head()

# %%
units = {
    "area_mean": "square pixels",
    "symmetry_mean": "unitless",
    "concavity_mean": "unitless",
    "compactness_mean": "unitless",
    "smoothness_mean": "unitless",
    "perimeter_mean": "pixels"
}

# Check for missing values
if df['concavity_mean'].isnull().values.any():
    print("Given column has NaN values.")
else:
    print("Given column does not have NaN values.")

if df['compactness_mean'].isnull().values.any():
    print("Given column has NaN values.")
else:
    print("Given column does not have NaN values.")

# %% [markdown]
# ## Appendix B — Descriptive Statistics

# %%
from utils import my_mean, my_median, my_mode, my_sample_SD, my_range, my_skewness

concavity = df['concavity_mean']
compactness = df['compactness_mean']

count_concavity = len(concavity)
count_compactness = len(compactness)

print("Concavity Count:", count_concavity)
print("Compactness Count:", count_compactness)

statistics = {
    'Concavity': {
        'Mean': my_mean(concavity),
        'Median': my_median(concavity),
        'Mode': my_mode(concavity),
        'Standard Deviation': my_sample_SD(concavity),
        'Range': my_range(concavity),
        'Skewness': my_skewness(concavity),
    },
    'Compactness': {
        'Mean': my_mean(compactness),
        'Median': my_median(compactness),
        'Mode': my_mode(compactness),
        'Standard Deviation': my_sample_SD(compactness),
        'Range': my_range(compactness),
        'Skewness': my_skewness(compactness),
    }
}

print(statistics)

# %% [markdown]
# ## Appendix C — Pearson's r & OLS Regression

# %%
def corr_scatter(column_x, column_y):
    """Calculate Pearson's r between two columns."""
    r = df[column_x].corr(df[column_y], method='pearson')
    return r

print("Pearson's R is equal to:", corr_scatter('compactness_mean', 'concavity_mean'))

# %%
def mult_regression(column_x, column_y):
    """
    Run OLS regression and produce diagnostic plots:
    scatter plot, QQ plot of residuals, histogram of residuals,
    residuals vs. fitted values.
    """
    if len(column_x) == 1:
        plt.figure()
        sns.regplot(
            x=column_x[0], y=column_y, data=df,
            marker="+", fit_reg=True, color='orange', ci=None
        )
        plt.xlabel(column_x[0] + "\n[" + units[column_x[0]] + "]")
        plt.ylabel(column_y + "\n[" + units[column_y] + "]")

    # Define predictors X and response Y
    X = df[column_x]
    X = sm.add_constant(X)
    Y = df[column_y]

    # Fit OLS model
    global regressionmodel
    regressionmodel = sm.OLS(Y, X).fit()
    print(regressionmodel.summary())

    residuals = regressionmodel.resid

    # QQ plot
    sm.qqplot(residuals, fit=True, line='45')
    plt.show()

    # Residuals histogram
    sns.histplot(residuals, kde=True, color="blue")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Residuals vs Fitted Values
    sns.residplot(
        x=regressionmodel.fittedvalues, y=residuals, color='green'
    )
    plt.xlabel("Fitted values for compactness_mean")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='black', linestyle='dashed')
    plt.show()


mult_regression(['compactness_mean'], 'concavity_mean')

# %% [markdown]
# ## Appendix D — Significance Testing & Confidence Interval

# %%
r = 0.883120670177251
x_mean = 0.10434098418277686
y_mean = 0.08879931581722322
sx = 0.052812757932512
sy = 0.0797198087078935
n = 569

# Slope
b1 = r * (sy / sx)
print("b1 =", round(b1, 3))

# Standard error of slope
SE = (sy / sx) * ((1 - r**2) / (n - 2)) ** 0.5
print("SE =", round(SE, 3))

# Critical t value for 95% CI
t95 = stats.t.ppf(0.975, n - 2)
print("t95 =", round(t95, 3))

# 95% Confidence interval
lower_bound = b1 - t95 * SE
upper_bound = b1 + t95 * SE
print("interval =", [round(lower_bound, 3), round(upper_bound, 3)])

# T-statistic
t_statistic = b1 / SE
print("t-statistic =", round(t_statistic, 3))

# One-tailed p-value
p_value = 1 - stats.t.cdf(t_statistic, df=n - 2)
print("p-value:", p_value)
