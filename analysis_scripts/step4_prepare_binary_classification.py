# -*- coding: utf-8 -*-
"""Step 4: Prepare data for binary classification."""

import os
import numpy as np
import pandas as pd

from step0_setup import DIRS, LOG_FILE
# Removed importing raw df_surv_tcga (use best-clustered file instead)
from step3_initial_clustering import best_model_info, df_expr_tcga
# Load best cluster assignments with Cluster label
best_clusters_file = os.path.join(DIRS['data'], 'tcga_best_clusters_full_features.csv')
df_surv_tcga_best = pd.read_csv(best_clusters_file, index_col='PatientID')

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print("\n--- 4. Preparing Data for Binary Classification ---")
cluster_dist = best_model_info["distribution"]
if not cluster_dist or len(cluster_dist) < 2:
    raise RuntimeError("Cannot perform binary classification with less than 2 clusters.")

sorted_clusters_by_size = sorted(cluster_dist.items(), key=lambda item: item[1], reverse=True)
top_two_clusters = [cluster_info[0] for cluster_info in sorted_clusters_by_size[:2]]
print(f"Selected clusters for binary classification (largest two): {top_two_clusters}")

# Subset the best-clustered survival DataFrame (with 'Cluster' column)
df_binary_subset = df_surv_tcga_best[df_surv_tcga_best["Cluster"].isin(top_two_clusters)].copy()
cluster_map = {top_two_clusters[0]: 0, top_two_clusters[1]: 1}
df_binary_subset["Label"] = df_binary_subset["Cluster"].map(cluster_map)
df_expr_binary = df_expr_tcga.loc[df_binary_subset.index]

X_full_binary = df_expr_binary.values
y_full_binary = df_binary_subset["Label"].values
feature_names_all = df_expr_binary.columns

binary_subset_file = os.path.join(DIRS["data"], "tcga_binary_classification_subset.csv")
df_binary_subset_save = pd.concat([
    df_binary_subset[['Cluster', 'Label', 'Overall Survival (Months)', 'Overall Survival Status']],
    df_expr_binary
], axis=1)
df_binary_subset_save.to_csv(binary_subset_file)
print(f"Saved binary classification subset data to: {binary_subset_file}")

with open(LOG_FILE, "a") as f:
    f.write("\nBinary Classification Data Preparation:\n")
    f.write(
        f"  Selected clusters: {top_two_clusters}\n  Subset size: {X_full_binary.shape[0]} patients, {X_full_binary.shape[1]} features\n"
    )
    f.write(
        f"  Label distribution (%): {dict(np.round(pd.Series(y_full_binary).value_counts(normalize=True)*100,1))}\n"
    )

print("Binary classification data preparation complete.")

# ------------------------------------------------------------------------
# Visualization: Kaplan-Meier & label distribution pie for binary targets
# ------------------------------------------------------------------------
print("\n--- 4a. Visualizing Binary Target Groups ---")
plt.figure(figsize=(12, 5))
# KM plot
ax1 = plt.subplot(1, 2, 1)
grp0 = df_binary_subset[df_binary_subset['Label'] == 0]
grp1 = df_binary_subset[df_binary_subset['Label'] == 1]
try:
    result = logrank_test(
        grp0['Overall Survival (Months)'], grp1['Overall Survival (Months)'],
        event_observed_A=grp0['Overall Survival Status'], event_observed_B=grp1['Overall Survival Status']
    )
    pval = result.p_value
except Exception:
    pval = float('nan')
kmf0, kmf1 = KaplanMeierFitter(), KaplanMeierFitter()
kmf0.fit(grp0['Overall Survival (Months)'], grp0['Overall Survival Status'], label=f"Class 0 (n={len(grp0)})")
kmf0.plot_survival_function(ax=ax1, ci_show=False)
kmf1.fit(grp1['Overall Survival (Months)'], grp1['Overall Survival Status'], label=f"Class 1 (n={len(grp1)})")
kmf1.plot_survival_function(ax=ax1, ci_show=False)
ax1.set_title(f"KM Survival of Classes (p={pval:.3g})")
ax1.set_xlabel('Survival Time (Months)')
ax1.set_ylabel('Survival Probability')
ax1.grid(True, linestyle='--', alpha=0.6)

# Pie chart
ax2 = plt.subplot(1, 2, 2)
counts = df_binary_subset['Label'].value_counts().sort_index()
ax2.pie(counts, labels=[f'Class {i}' for i in counts.index], autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
ax2.set_title('Label Distribution')
plt.axis('equal')

vis_path = os.path.join(DIRS['plots_classification'], 'binary_target_groups.png')
plt.tight_layout(pad=2.0)
plt.savefig(vis_path, bbox_inches='tight', dpi=150)
print(f"Saved binary target visualization to: {vis_path}")
with open(LOG_FILE, 'a') as f:
    f.write(f"  Saved binary target viz: {vis_path}\n")
plt.show()
