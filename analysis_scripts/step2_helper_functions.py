# -*- coding: utf-8 -*-
"""Step 2: Helper function definitions for the analysis."""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from step0_setup import LOG_FILE


def run_multi_group_survival(df_survival_with_clusters, cluster_col="Cluster",
                             time_col="Overall Survival (Months)", event_col="Overall Survival Status",
                             plot_title="Kaplan-Meier Curves by Cluster", out_prefix=None, show_plot=True):
    """Performs multi-group KM analysis and pairwise log-rank tests."""
    unique_clusters = sorted(df_survival_with_clusters[cluster_col].unique())
    n_groups = len(unique_clusters)

    if n_groups < 2:
        print(f"Warning: Only {n_groups} cluster found. Cannot perform survival comparison.")
        return 1.0, {}, "Not enough clusters for comparison.", {}

    cluster_distribution = df_survival_with_clusters[cluster_col].value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n_groups))
    medians = {}

    for idx, cluster in enumerate(unique_clusters):
        group = df_survival_with_clusters[df_survival_with_clusters[cluster_col] == cluster]
        if group.empty:
            print(f"Warning: Cluster {cluster} has no samples. Skipping.")
            continue
        if group[event_col].sum() == 0:
            print(f"Warning: Cluster {cluster} has no observed events. KM curve might be misleading.")

        kmf = KaplanMeierFitter()
        try:
            kmf.fit(durations=group[time_col], event_observed=group[event_col],
                    label=f"Cluster {cluster} (n={len(group)})")
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[idx])
            medians[cluster] = kmf.median_survival_time_
            if np.isfinite(medians[cluster]):
                ax.axvline(medians[cluster], linestyle="--", color=colors[idx], alpha=0.7, ymax=0.5)
        except Exception as e:
            print(f"Error fitting KM for Cluster {cluster}: {e}")
            medians[cluster] = np.nan

    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel("Survival Time (Months)")
    ax.set_ylabel("Survival Probability")
    ax.legend(title="Clusters", loc="best")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    signif_text = "Pairwise Log-rank Tests:\n"
    all_p_values = []
    for (c1, c2) in combinations(unique_clusters, 2):
        group1 = df_survival_with_clusters[df_survival_with_clusters[cluster_col] == c1]
        group2 = df_survival_with_clusters[df_survival_with_clusters[cluster_col] == c2]
        if not group1.empty and not group2.empty and group1[event_col].sum() > 0 and group2[event_col].sum() > 0:
            try:
                result = logrank_test(group1[time_col], group2[time_col],
                                       event_observed_A=group1[event_col], event_observed_B=group2[event_col])
                p_val = result.p_value
                all_p_values.append(p_val)
                star = "**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")
                signif_text += f" C{c1} vs C{c2}: p={p_val:.3g}{star}\n"
            except Exception as e:
                print(f"Error during logrank test for {c1} vs {c2}: {e}")
                signif_text += f" C{c1} vs C{c2}: Error\n"
        else:
            signif_text += f" C{c1} vs C{c2}: Skipped (no data/events)\n"

    min_p = np.min(all_p_values) if all_p_values else 1.0
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.02, 0.5, signif_text.strip(), transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=props)

    if out_prefix:
        plot_filename = f"{out_prefix}_km_plot.png"
        try:
            plt.savefig(plot_filename, bbox_inches="tight", dpi=150)
            print(f"Saved KM plot: {plot_filename}")
            with open(LOG_FILE, "a") as f:
                f.write(f"  Saved plot: {plot_filename}\n")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return min_p, cluster_distribution, signif_text, medians


def cluster_and_evaluate(X, df_surv, method_name="kmeans", k=2, linkage="ward",
                         time_col="Overall Survival (Months)", event_col="Overall Survival Status"):
    """Cluster data and evaluate survival separation."""
    print(f"  Applying {method_name} with k={k}...")
    model = None
    clusters = None
    try:
        if method_name.lower() == "kmeans":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = model.fit_predict(X)
        elif method_name.lower() == "agglomerative":
            if X.shape[0] > 5000:
                print(f"Warning: Large dataset ({X.shape[0]}) for AgglomerativeClustering.")
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            clusters = model.fit_predict(X)
        elif method_name.lower() == "gmm":
            model = GaussianMixture(n_components=k, random_state=42, n_init=5)
            clusters = model.fit_predict(X)
        else:
            raise ValueError(f"Unknown clustering method: {method_name}")

        df_tmp = df_surv.copy()
        df_tmp["Cluster"] = clusters
        print(f"  Evaluating survival separation for k={k}...")
        min_p, distribution, signif_text, medians = run_multi_group_survival(
            df_tmp, cluster_col="Cluster", time_col=time_col, event_col=event_col,
            plot_title=f"{method_name} k={k} Survival (Evaluation)",
            out_prefix=None, show_plot=False)
        print(f"  k={k}: min p-value = {min_p:.4f}, Distribution = {distribution}")
        return min_p, distribution, clusters, model, signif_text, medians
    except Exception as e:
        print(f"Error during clustering/evaluation for {method_name} k={k}: {e}")
        return 1.0, {}, np.array([]), None, f"Error: {e}", {}


def interpolate_roc(fpr, tpr, n_points=100):
    """Interpolate TPR onto a common FPR grid."""
    fpr_grid = np.linspace(0, 1, n_points)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return fpr_grid, tpr_interp


def plot_roc_curves_all_models(roc_storage, models_to_plot, feature_sizes,
                               output_file, highlight_best=None):
    """Plot mean ROC curves with variability bands."""
    num_models = len(models_to_plot)
    fig, axes = plt.subplots(1, num_models, figsize=(5.5 * num_models, 5.5), sharey=True, sharex=True)
    if num_models == 1:
        axes = [axes]
    fs_colors = plt.cm.viridis(np.linspace(0, 1, len(feature_sizes)))
    fs_color_map = {fs: fs_colors[i] for i, fs in enumerate(sorted(feature_sizes))}
    common_fpr_grid = np.linspace(0, 1, 100)

    for ax, model_name in zip(axes, models_to_plot):
        ax.set_title(f"{model_name}", fontsize=14)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        if ax == axes[0]:
            ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.6, label="Chance")

        for fsize in sorted(feature_sizes):
            run_data_list = roc_storage.get((model_name, fsize), [])
            if not run_data_list:
                continue
            tprs_interp = []
            aucs = []
            for run_data in run_data_list:
                if (run_data and 'fpr' in run_data and 'tpr' in run_data and 'auc' in run_data and
                        not np.isnan(run_data['auc']) and isinstance(run_data['fpr'], np.ndarray) and
                        isinstance(run_data['tpr'], np.ndarray) and len(run_data['fpr']) > 1):
                    _, tpr_i = interpolate_roc(run_data['fpr'], run_data['tpr'], n_points=len(common_fpr_grid))
                    tprs_interp.append(tpr_i)
                    aucs.append(run_data['auc'])
            if not tprs_interp:
                continue

            mean_tpr = np.mean(tprs_interp, axis=0)
            std_tpr = np.std(tprs_interp, axis=0)
            mean_auc = np.mean(aucs)
            color = fs_color_map.get(fsize, 'black')
            is_highlighted = highlight_best and (model_name, fsize) == highlight_best
            lw, zorder = (2.5, 3) if is_highlighted else (1.5, 2)
            label = f"f={fsize} (AUC={mean_auc:.3f})" + (" *" if is_highlighted else "")
            ax.plot(common_fpr_grid, mean_tpr, color=color, lw=lw, label=label, zorder=zorder)
            ax.fill_between(common_fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=color, alpha=0.2, zorder=zorder - 1)

        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Mean ROC Curves by Model and Feature Size (± Std Dev across Runs)",
                 y=1.03, fontsize=16)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    try:
        plt.savefig(output_file, bbox_inches="tight", dpi=150)
        print(f"Saved ROC curve plot to: {output_file}")
        with open(LOG_FILE, "a") as f:
            f.write(f"  Saved ROC curve plot: {output_file}\n")
    except Exception as e:
        print(f"Error saving ROC plot {output_file}: {e}")
    plt.show()

print("Helper functions defined.")
