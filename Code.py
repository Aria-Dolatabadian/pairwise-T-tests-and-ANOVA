import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, f_oneway, shapiro, levene, wilcoxon, kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
# Read data from CSV
data_read = pd.read_csv('sample_data.csv')

# Perform pairwise T-tests
results_ttest = {}
for i, col1 in enumerate(data_read.columns):
    for j, col2 in enumerate(data_read.columns):
        if i < j:
            t_stat, p_val = ttest_rel(data_read[col1], data_read[col2])
            results_ttest[f'{col1} vs {col2}'] = {'T-statistic': t_stat, 'P-value': p_val}

# Print pairwise T-test results
print("Pairwise T-test Results:")
for comparison, result in results_ttest.items():
    print(f"{comparison}: T-statistic = {result['T-statistic']}, P-value = {result['P-value']}")


# Visualize results - Box plot
data_read.boxplot()
plt.ylabel('Value')
plt.title('Boxplot of Sample Data')
plt.show()

# Visualize pairwise T-test results
plt.bar(range(len(results_ttest)), [result['P-value'] for result in results_ttest.values()], tick_label=list(results_ttest.keys()))
plt.xticks(rotation=90)
plt.xlabel('Comparison')
plt.ylabel('P-value')
plt.title('Pairwise T-test P-values')
plt.show()



# Read data from CSV
data_read = pd.read_csv('sample_data.csv')

# Perform normality test
normality_results = {}
for col in data_read.columns:
    stat, p_val = shapiro(data_read[col])
    normality_results[col] = {'Test Statistic': stat, 'P-value': p_val}

# Print normality test results
print("Normality Test Results:")
for col, result in normality_results.items():
    print(f"{col}: Test Statistic = {result['Test Statistic']}, P-value = {result['P-value']}")

# Perform ANOVA
f_stat, p_val_anova = f_oneway(*[data_read[col] for col in data_read.columns])

# Print ANOVA results
print(f"\nANOVA Results:\nF-statistic = {f_stat}, P-value = {p_val_anova}")

# Check homogeneity of variances
stat, p_val_levene = levene(*[data_read[col] for col in data_read.columns])

# Print Levene's test results
print(f"\nLevene's Test for Homogeneity of Variance:\nTest Statistic = {stat}, P-value = {p_val_levene}")

# Perform post hoc tests (Tukey's HSD)
posthoc_results = {}
if p_val_anova < 0.05:  # Proceed with post hoc tests only if ANOVA is significant
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(np.concatenate([data_read[col] for col in data_read.columns]),
                              np.repeat(data_read.columns, len(data_read)))
    posthoc_results['Tukey HSD'] = tukey.summary()

    # Print post hoc results
    print("\nTukey's HSD Post Hoc Test Results:")
    print(tukey)

# Calculate effect sizes
effect_sizes = {}
for col1 in data_read.columns:
    for col2 in data_read.columns:
        if col1 != col2:
            pooled_std = np.sqrt(((len(data_read[col1]) - 1) * np.var(data_read[col1], ddof=1) +
                                  (len(data_read[col2]) - 1) * np.var(data_read[col2], ddof=1)) /
                                 (len(data_read[col1]) + len(data_read[col2]) - 2))
            effect_sizes[f'{col1} vs {col2}'] = np.abs(data_read[col1].mean() - data_read[col2].mean()) / pooled_std

# Print effect sizes
print("\nEffect Sizes:")
for comparison, effect_size in effect_sizes.items():
    print(f"{comparison}: Effect Size = {effect_size}")

# Visualize results - Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_read)
plt.ylabel('Value')
plt.title('Boxplot of Sample Data')
plt.show()

# Perform non-parametric tests (Wilcoxon signed-rank test)
non_parametric_results = {}
for col1 in data_read.columns:
    for col2 in data_read.columns:
        if col1 != col2:
            stat, p_val = wilcoxon(data_read[col1], data_read[col2])
            non_parametric_results[f'{col1} vs {col2}'] = {'Test Statistic': stat, 'P-value': p_val}

# Print non-parametric test results
print("\nWilcoxon Signed-Rank Test Results:")
for comparison, result in non_parametric_results.items():
    print(f"{comparison}: Test Statistic = {result['Test Statistic']}, P-value = {result['P-value']}")

# Perform non-parametric test for overall difference (Kruskal-Wallis test)
stat, p_val_kruskal = kruskal(*[data_read[col] for col in data_read.columns])

# Print Kruskal-Wallis test results
print(f"\nKruskal-Wallis Test for Overall Difference:\nTest Statistic = {stat}, P-value = {p_val_kruskal}")

# Multiple comparison correction for pairwise T-tests
ttest_p_vals = [result['P-value'] for result in non_parametric_results.values()]
reject, pvals_corrected, _, _ = multipletests(ttest_p_vals, method='bonferroni')

# Print corrected p-values
print("\nCorrected P-values (Bonferroni Correction):")
for comparison, p_val_corrected, reject_ in zip(non_parametric_results.keys(), pvals_corrected, reject):
    print(f"{comparison}: Corrected P-value = {p_val_corrected}, Reject Null Hypothesis = {reject_}")

# Visualize results - Pairwise T-test corrected P-values
plt.figure(figsize=(10, 6))
plt.bar(range(len(non_parametric_results)), pvals_corrected, tick_label=list(non_parametric_results.keys()))
plt.xticks(rotation=90)
plt.xlabel('Comparison')
plt.ylabel('Corrected P-value')
plt.title('Pairwise T-test Corrected P-values (Bonferroni Correction)')
plt.show()
