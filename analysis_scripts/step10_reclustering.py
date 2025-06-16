# -*- coding: utf-8 -*-
"""Step 10: Re-cluster using selected features."""

import os
import pandas as pd

from step0_setup import DIRS, LOG_FILE
from step1_load_tcga import df_expr_tcga, df_surv_tcga
from step2_helper_functions import cluster_and_evaluate, run_multi_group_survival
from step3_initial_clustering import best_model_info
from step8_select_best_model import selected_features_final_combined

print("\n--- 10. Re-clustering with Selected Features ---")

if not isinstance(selected_features_final_combined, (list, pd.Index)):
    raise ValueError("selected_features_final_combined is not a valid sequence")
if len(selected_features_final_combined) == 0:
    raise ValueError("selected_features_final_combined is empty")

method = best_model_info.get("method")
best_k = best_model_info.get("k")
if method is None or best_k is None:
    raise RuntimeError("Best clustering configuration from step 3 is unavailable")

print(f"Using best method from initial clustering: {method} (k={best_k})")
print(f"Number of selected features from step 8: {len(selected_features_final_combined)}")

X_sel = df_expr_tcga[selected_features_final_combined].values

min_p, distribution, clusters, model, signif_text, medians = cluster_and_evaluate(
    X=X_sel,
    df_surv=df_surv_tcga,
    method_name=method,
    k=best_k,
)

df_surv_tcga_reclust = df_surv_tcga.loc[df_expr_tcga.index].copy()
df_surv_tcga_reclust["Cluster"] = clusters

cluster_file = os.path.join(DIRS["data"], "tcga_reclustering_selected_features.csv")
df_surv_tcga_reclust.to_csv(cluster_file)
print(f"Saved re-clustered assignments to: {cluster_file}")

evaluation_df = pd.DataFrame([
    {
        "method": method,
        "k": best_k,
        "min_p_value": min_p,
        "distribution": distribution,
    }
])
eval_file = os.path.join(DIRS["results"], "reclustering_evaluation.csv")
evaluation_df.to_csv(eval_file, index=False)
print(f"Saved KM evaluation results to: {eval_file}")

print("Generating KM plot for the re-clustered data...")
plot_prefix = os.path.join(DIRS["plots_clustering"], f"tcga_recluster_{method}_k{best_k}")
run_multi_group_survival(
    df_surv_tcga_reclust,
    cluster_col="Cluster",
    plot_title=f"TCGA Survival by Re-Clustering ({method}, k={best_k}, p={min_p:.3g})",
    out_prefix=plot_prefix,
    show_plot=True,
)

with open(LOG_FILE, "a") as f:
    f.write("\nRe-clustering with Selected Features:\n")
    f.write(f"  Method: {method}, k: {best_k}, Min p: {min_p:.4g}\n")
    f.write(f"  Cluster distribution: {distribution}\n")
    f.write(f"  Saved assignments: {cluster_file}\n  Saved evaluation: {eval_file}\n")

print("Re-clustering complete.")
