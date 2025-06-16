# -*- coding: utf-8 -*-
"""Step 3: Initial clustering of TCGA data."""

import os
import pickle
import pandas as pd

from step0_setup import DIRS, CLUSTERING_METHODS, K_RANGE, LOG_FILE
from step1_load_tcga import df_expr_tcga, df_surv_tcga
from step2_helper_functions import cluster_and_evaluate, run_multi_group_survival

print("\n--- 3. Starting Initial Clustering Analysis (TCGA, All Features) ---")
clustering_results = []
best_model_info = {
    "method": None,
    "k": None,
    "min_p_value": 1.0,
    "distribution": None,
    "clusters": None,
    "model": None,
    "signif_text": "",
    "medians": {}
}

for method in CLUSTERING_METHODS:
    print(f"\n--- Evaluating Method: {method} ---")
    best_k_for_method = None
    min_p_for_method = 1.0
    for k in K_RANGE:
        min_p, distribution, clusters, model, signif_text, medians = cluster_and_evaluate(
            X=df_expr_tcga.values,
            df_surv=df_surv_tcga,
            method_name=method,
            k=k,
        )
        clustering_results.append({"method": method, "k": k, "min_p_value": min_p, "distribution": distribution})
        if min_p < min_p_for_method:
            min_p_for_method = min_p
            best_k_for_method = k
        if min_p < best_model_info["min_p_value"]:
            print(f"  * New best found: {method} k={k}, p={min_p:.4f}")
            best_model_info.update({
                "method": method,
                "k": k,
                "min_p_value": min_p,
                "distribution": distribution,
                "clusters": clusters,
                "model": model,
                "signif_text": signif_text,
                "medians": medians,
            })
    print(f"--- Best k for {method}: {best_k_for_method} (p={min_p_for_method:.4f}) ---")

# Save evaluation results
results_df = pd.DataFrame(clustering_results)
results_file = os.path.join(DIRS["results"], "initial_clustering_evaluation.csv")
results_df.to_csv(results_file, index=False)
print(f"\nSaved clustering evaluation results to: {results_file}")

print("\n--- Overall Best Clustering Configuration (Full Features) ---")
if best_model_info["method"]:
    print(
        f"Method: {best_model_info['method']}, k={best_model_info['k']}, Min p-value: {best_model_info['min_p_value']:.4g}"
    )
    print(f"Distribution: {best_model_info['distribution']}")
    df_surv_tcga_best_clusters = df_surv_tcga.loc[df_expr_tcga.index].copy()
    df_surv_tcga_best_clusters["Cluster"] = best_model_info["clusters"]
    best_clusters_file = os.path.join(DIRS["data"], "tcga_best_clusters_full_features.csv")
    df_surv_tcga_best_clusters.to_csv(best_clusters_file)
    print(f"Saved best cluster assignments to: {best_clusters_file}")
    best_model_file = os.path.join(DIRS["models_clustering"], "best_clustering_model.pkl")
    with open(best_model_file, "wb") as f:
        pickle.dump(best_model_info["model"], f)
    print(f"Saved best clustering model object to: {best_model_file}")

    print("Generating KM plot for the best clustering configuration...")
    plot_prefix = os.path.join(
        DIRS["plots_clustering"], f"tcga_best_{best_model_info['method']}_k{best_model_info['k']}"
    )
    run_multi_group_survival(
        df_surv_tcga_best_clusters,
        cluster_col="Cluster",
        plot_title=(
            f"TCGA Survival by Best Clustering ({best_model_info['method']}, k={best_model_info['k']}, p={best_model_info['min_p_value']:.3g})"
        ),
        out_prefix=plot_prefix,
        show_plot=True,
    )
    with open(LOG_FILE, "a") as f:
        f.write("\nBest Initial Clustering (TCGA, Full Features):\n")
        f.write(
            f"  Method: {best_model_info['method']}, k: {best_model_info['k']}, Min p: {best_model_info['min_p_value']:.4g}\n"
        )
        f.write(f"  Saved assignments: {best_clusters_file}\n  Saved model: {best_model_file}\n")
else:
    print("Error: No suitable clustering configuration found.")
    with open(LOG_FILE, "a") as f:
        f.write("\nError: No suitable initial clustering configuration found.\n")
    raise RuntimeError("Clustering analysis failed.")

print("\nInitial clustering analysis complete.")
