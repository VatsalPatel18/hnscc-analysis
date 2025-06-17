# -*- coding: utf-8 -*-
"""Step 6: Aggregate results and generate visualizations."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from step0_setup import DIRS, CLASSIFIERS, FEATURE_SIZES, LOG_FILE
from step5_classification_loop import results_df, roc_data_storage
from step2_helper_functions import plot_roc_curves_all_models

print("\n--- 6. Aggregating and Visualizing Classification Results ---")

df_agg = results_df.groupby(["FeatureSize", "Model"])["AUC"].agg(["mean", "std"]).reset_index()
agg_results_csv = os.path.join(DIRS["results"], "classification_metrics_aggregated.csv")
df_agg.to_csv(agg_results_csv, index=False)
print(f"Saved aggregated classification metrics to: {agg_results_csv}")

print("Generating AUC boxplots...")
model_list = sorted(results_df["Model"].unique())
num_models = len(model_list)
sorted_sizes = sorted(results_df["FeatureSize"].unique())
fig_box, axes_box = plt.subplots(1, num_models, figsize=(4.5 * num_models, 5), sharey=True)
if num_models == 1:
    axes_box = [axes_box]
for ax, model in zip(axes_box, model_list):
    df_model = results_df[results_df["Model"] == model]
    data_for_boxplot = []
    xtick_labels = []
    for fs in sorted_sizes:
        if fs in df_model["FeatureSize"].values:
            auc_values = df_model[df_model["FeatureSize"] == fs]["AUC"].dropna().values
            if len(auc_values) > 0:
                data_for_boxplot.append(auc_values)
                xtick_labels.append(str(fs))
    if data_for_boxplot:
        bp = ax.boxplot(data_for_boxplot, patch_artist=True, showmeans=True, labels=xtick_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_boxplot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"{model}", fontsize=12)
        ax.set_xlabel("Feature Size")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_title(f"{model}\n(No Data)", fontsize=12)
        ax.set_xlabel("Feature Size")
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    if ax == axes_box[0]:
        ax.set_ylabel("AUC (Test Set)")
plt.suptitle("AUC Distribution across Runs by Feature Size", y=1.02, fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
boxplot_file = os.path.join(DIRS["plots_classification"], "auc_boxplots_by_model.png")
plt.savefig(boxplot_file, bbox_inches="tight", dpi=150)
print(f"Saved AUC boxplots to: {boxplot_file}")
plt.close()

print("Generating Mean AUC line plots...")
plt.figure(figsize=(10, 7))
markers = ['o', 's', '^', 'd', 'v', '*', 'p']
colors = plt.cm.tab10(np.linspace(0, 1, len(model_list)))
best_fsize_auc = None
best_model_auc = None
best_mean_auc = None
if not df_agg.empty:
    df_agg_valid = df_agg.dropna(subset=['mean'])
    if not df_agg_valid.empty:
        best_row = df_agg_valid.loc[df_agg_valid["mean"].idxmax()]
        best_fsize_auc = best_row["FeatureSize"]
        best_model_auc = best_row["Model"]
        best_mean_auc = best_row["mean"]
        print(f"\nBest Performance (Mean AUC only): Model={best_model_auc}, FSize={best_fsize_auc}, Mean AUC={best_mean_auc:.4f}")
    else:
        print("\nWarning: No valid mean AUC values found for aggregation.")

for i, model in enumerate(model_list):
    data_model = df_agg[df_agg["Model"] == model].sort_values("FeatureSize")
    if not data_model.empty:
        x = data_model["FeatureSize"].values
        y_mean = data_model["mean"].values
        y_std = data_model["std"].fillna(0).values
        plt.plot(x, y_mean, label=model, marker=markers[i % len(markers)], linestyle="-", color=colors[i])
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15, color=colors[i])
if best_model_auc:
    plt.scatter(best_fsize_auc, best_mean_auc, marker="*", s=250, color="blue", zorder=5, label=f"Best AUC: {best_model_auc} @ {int(best_fsize_auc)}")
    plt.text(best_fsize_auc * 1.02, best_mean_auc, f" {best_mean_auc:.3f}", fontsize=9, color="blue", verticalalignment='center')

plt.xlabel("Number of Selected Features")
plt.ylabel("Mean AUC (Test Set ± Std Dev)")
plt.title("Classifier Performance vs. Number of Features")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(bottom=0)
plt.xticks(FEATURE_SIZES, labels=[str(fs) for fs in FEATURE_SIZES], rotation=45, ha='right')
lineplot_file = os.path.join(DIRS["plots_classification"], "auc_mean_std_lines_by_model.png")
plt.savefig(lineplot_file, bbox_inches="tight", dpi=150)
print(f"Saved Mean AUC line plot to: {lineplot_file}")
plt.close()

print("Generating ROC plots...")
roc_plot_file = os.path.join(DIRS["plots_classification"], "roc_curves_by_model.png")
models_to_plot = sorted(list(CLASSIFIERS.keys()))
highlight_auc_only = (best_model_auc, best_fsize_auc) if best_model_auc else None
plot_roc_curves_all_models(roc_storage=roc_data_storage, models_to_plot=models_to_plot, feature_sizes=FEATURE_SIZES, output_file=roc_plot_file, highlight_best=highlight_auc_only)

with open(LOG_FILE, "a") as f:
    f.write("\nClassification Visualization Completed:\n")
    f.write(f"  Saved aggregated metrics: {agg_results_csv}\n")
    f.write(f"  Saved AUC boxplots: {boxplot_file}\n  Saved Mean AUC line plot: {lineplot_file}\n")
    f.write(f"  Saved ROC curve plot: {roc_plot_file}\n")
    if best_model_auc:
        f.write(f"  Overall Best Combo (Mean AUC only): Model={best_model_auc}, FSize={best_fsize_auc}, AUC={best_mean_auc:.4f}\n")

print("\nAggregation and visualization complete.")
