# -*- coding: utf-8 -*-
"""Step 4: Prepare data for binary classification."""

import os
import numpy as np
import pandas as pd

from step0_setup import DIRS, LOG_FILE
from step3_initial_clustering import best_model_info, df_surv_tcga, df_expr_tcga

print("\n--- 4. Preparing Data for Binary Classification ---")
cluster_dist = best_model_info["distribution"]
if not cluster_dist or len(cluster_dist) < 2:
    raise RuntimeError("Cannot perform binary classification with less than 2 clusters.")

sorted_clusters_by_size = sorted(cluster_dist.items(), key=lambda item: item[1], reverse=True)
top_two_clusters = [cluster_info[0] for cluster_info in sorted_clusters_by_size[:2]]
print(f"Selected clusters for binary classification (largest two): {top_two_clusters}")

df_binary_subset = df_surv_tcga[df_surv_tcga["Cluster"].isin(top_two_clusters)].copy()
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
