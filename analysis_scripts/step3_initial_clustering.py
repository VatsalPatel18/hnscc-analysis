# -*- coding: utf-8 -*-
"""Step 3: Initial clustering of TCGA data."""

import os
import pickle
import pandas as pd

from step0_setup import DIRS, CLUSTERING_METHODS, K_RANGE, LOG_FILE
from step1_load_tcga import df_expr_tcga, df_surv_tcga
from step2_helper_functions import cluster_and_evaluate, run_multi_group_survival
# For comprehensive clustering metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

print("\n--- 3a. Full Clustering Evaluation Metrics (Silhouette, CH, DB, Inertia) ---")
full_results = []
for method in CLUSTERING_METHODS:
    for k in K_RANGE:
        try:
            # instantiate clustering model
            if method.lower() == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif method.lower() == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            elif method.lower() == 'gmm':
                model = GaussianMixture(n_components=k, random_state=42, n_init=5)
            else:
                continue
            labels = model.fit_predict(df_expr_tcga.values)
            # compute ML metrics
            if len(set(labels)) > 1:
                sil = silhouette_score(df_expr_tcga.values, labels)
                ch = calinski_harabasz_score(df_expr_tcga.values, labels)
                db = davies_bouldin_score(df_expr_tcga.values, labels)
            else:
                sil = ch = db = float('nan')
            inertia = model.inertia_ if hasattr(model, 'inertia_') else float('nan')
            full_results.append({
                'Method': method, 'k': k,
                'Silhouette': sil, 'CalinskiHarabasz': ch,
                'DaviesBouldin': db, 'Inertia': inertia
            })
            print(f"    {method} k={k}: Silhouette={sil:.3f}, CH={ch:.3f}, DB={db:.3f}, Inertia={inertia:.3f}")
        except Exception as e:
            print(f"    Error computing metrics for {method} k={k}: {e}")

df_full = pd.DataFrame(full_results)
full_file = os.path.join(DIRS['results'], 'full_clustering_evaluation.csv')
df_full.to_csv(full_file, index=False)
print(f"Saved full clustering evaluation metrics to: {full_file}")
try:
    fig, ax = plt.subplots()
    df_km = df_full[df_full['Method'].str.lower()=='kmeans']
    ax.plot(df_km['k'], df_km['Inertia'], marker='o')
    ax.set_title('KMeans Inertia vs k')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.grid(True)
    elbow_plot = os.path.join(DIRS['plots_clustering'], 'kmeans_elbow.png')
    fig.tight_layout()
    fig.savefig(elbow_plot, bbox_inches='tight', dpi=150)
    print(f"Saved KMeans elbow plot to: {elbow_plot}")
except Exception as e:
    print(f"Error plotting KMeans elbow: {e}")

print("\n--- 3b. Starting Initial Clustering Analysis (TCGA, All Features) ---")
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
