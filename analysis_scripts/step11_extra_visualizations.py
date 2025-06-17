# -*- coding: utf-8 -*-
"""Step 11: Generate additional plots for classification results and selected features."""

import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from upsetplot import UpSet, from_memberships
    upsetplot_available = True
except ImportError:
    print("WARNING: upsetplot not installed, skipping UpSet plots.")
    upsetplot_available = False

from step0_setup import DIRS, FEATURE_SIZES, CLASSIFIERS, LOG_FILE


ROC_DATA_FILE = os.path.join(DIRS["results"], "roc_data_all_runs.pkl")
METRICS_CSV = os.path.join(DIRS["results"], "classification_metrics_all_runs.csv")
FEATURE_DIR = DIRS["results_features"]


# ---------------------------------------------------------------------------
# ROC curves across feature sizes
# ---------------------------------------------------------------------------

def plot_mean_roc_feature_grid(out_file, feature_sizes=None, models=None):
    """Plot mean ROC curves for each feature size in a grid of subplots."""
    feature_sizes = feature_sizes or FEATURE_SIZES
    models = models or list(CLASSIFIERS.keys())

    if not os.path.exists(ROC_DATA_FILE):
        print(f"ROC data file not found: {ROC_DATA_FILE}")
        return
    with open(ROC_DATA_FILE, "rb") as f:
        roc_storage = pickle.load(f)

    n_sizes = len(feature_sizes)
    n_cols = 2
    n_rows = int(np.ceil(n_sizes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5 * n_rows), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    common_fpr = np.linspace(0, 1, 100)
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(models)))

    for idx, fsize in enumerate(feature_sizes):
        ax = axes_flat[idx]
        for j, model_name in enumerate(models):
            runs = roc_storage.get((model_name, fsize), [])
            tprs, aucs = [], []
            for run in runs:
                fpr = run.get("fpr")
                tpr = run.get("tpr")
                auc = run.get("auc")
                if isinstance(fpr, np.ndarray) and len(fpr) > 1 and not np.isnan(auc):
                    tprs.append(np.interp(common_fpr, fpr, tpr))
                    aucs.append(auc)
            if not tprs:
                continue
            tprs = np.array(tprs)
            mean_tpr = tprs.mean(axis=0)
            std_tpr = tprs.std(axis=0)
            mean_auc = np.mean(aucs)
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
            ax.fill_between(common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=colors[j], alpha=0.25)
            ax.plot(common_fpr, mean_tpr, lw=2, color=colors[j],
                    label=f"{model_name} (AUC={mean_auc:.3f})")
        ax.set_title(f"Feature Size f={fsize}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=7)

    for ax in axes_flat[-n_cols:]:
        ax.set_xlabel("False Positive Rate")
    for ax in axes_flat[::n_cols]:
        ax.set_ylabel("True Positive Rate")

    fig.suptitle("Mean ROC Curves \u00b1 Std Dev Across Models for Various Feature Sizes",
                 fontsize=16, y=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved ROC grid plot to: {out_file}")
    with open(LOG_FILE, "a") as f:
        f.write(f"  Saved ROC grid plot: {out_file}\n")


# ---------------------------------------------------------------------------
# Boxplot of AUC by feature size and model
# ---------------------------------------------------------------------------

def plot_auc_boxplot(out_file):
    if not os.path.exists(METRICS_CSV):
        print(f"Metrics CSV not found: {METRICS_CSV}")
        return
    df = pd.read_csv(METRICS_CSV)
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="FeatureSize", y="AUC", hue="Model", palette=palette)
    plt.title("AUC Distribution by Feature Size and Model")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Test-set AUC")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved AUC boxplot to: {out_file}")
    with open(LOG_FILE, "a") as f:
        f.write(f"  Saved AUC boxplot: {out_file}\n")


# ---------------------------------------------------------------------------
# Determine best run per feature size using AUC
# ---------------------------------------------------------------------------

def load_best_feature_sets():
    """Return a mapping of label -> list of genes for the best run of each feature size."""
    if not os.path.exists(METRICS_CSV):
        print(f"Metrics CSV not found: {METRICS_CSV}")
        return {}
    df = pd.read_csv(METRICS_CSV)
    best_runs = {}
    for fsize in df["FeatureSize"].unique():
        df_fs = df[df["FeatureSize"] == fsize].dropna(subset=["AUC"])
        if df_fs.empty:
            continue
        best_row = df_fs.loc[df_fs["AUC"].idxmax()]
        best_runs[fsize] = int(best_row["Run"])

    feature_sets = {}
    for fsize, run in best_runs.items():
        fname = os.path.join(FEATURE_DIR, f"selected_features_f{fsize}_run{run}.txt")
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            genes = [line.strip() for line in f if line.strip()]
        feature_sets[f"f{fsize}"] = genes
    return feature_sets


# ---------------------------------------------------------------------------
# UpSet plot and heatmap for best feature sets
# ---------------------------------------------------------------------------

def plot_upset(feature_sets, out_file):
    if not upsetplot_available:
        print("Skipping UpSet plot (upsetplot not available).")
        return
    if not feature_sets:
        print("No feature sets provided for UpSet plot.")
        return
    membership_map = {}
    for label, genes in feature_sets.items():
        for gene in genes:
            membership_map.setdefault(gene, []).append(label)
    memberships = from_memberships(membership_map.values())
    upset = UpSet(memberships, subset_size='count', show_counts=True, sort_by='degree')
    upset.plot()
    plt.suptitle("Feature Overlap Across Best Model Runs", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved UpSet plot to: {out_file}")
    with open(LOG_FILE, "a") as f:
        f.write(f"  Saved UpSet plot: {out_file}\n")


def plot_feature_heatmap(feature_sets, out_file):
    if not feature_sets:
        print("No feature sets provided for heatmap.")
        return
    all_genes = sorted({g for genes in feature_sets.values() for g in genes})
    data = []
    for gene in all_genes:
        row = [1 if gene in feature_sets[label] else 0 for label in feature_sets]
        data.append(row)
    df = pd.DataFrame(data, index=all_genes, columns=list(feature_sets.keys()))
    plt.figure(figsize=(10, max(6, 0.3 * len(all_genes))))
    sns.heatmap(df, cmap="Greens", cbar=False, linewidths=0.5, linecolor='lightgray')
    plt.title("Gene Occurrence Across Best Model Runs")
    plt.xlabel("Feature Size")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved heatmap to: {out_file}")
    with open(LOG_FILE, "a") as f:
        f.write(f"  Saved heatmap: {out_file}\n")


# ---------------------------------------------------------------------------
# Frequency plots across all runs
# ---------------------------------------------------------------------------

def plot_feature_frequencies(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_files = os.listdir(FEATURE_DIR)
    pat = re.compile(r"^selected_features_f(\d+)_run(\d+)\.txt$")
    file_info = []
    for fname in all_files:
        m = pat.match(fname)
        if m:
            fsize = int(m.group(1))
            run = int(m.group(2))
            file_info.append((fsize, run, fname))
    feature_sizes = sorted({f for f, _, _ in file_info})
    for fsize in feature_sizes:
        runs = [(run, fname) for f, run, fname in file_info if f == fsize]
        n_runs = len(runs)
        if n_runs == 0:
            continue
        counter = Counter()
        for run, fname in runs:
            path = os.path.join(FEATURE_DIR, fname)
            with open(path) as f:
                feats = [line.strip() for line in f if line.strip()]
            counter.update(feats)
        df = (pd.DataFrame.from_dict(counter, orient="index", columns=["Count"]) \
              .assign(Frequency=lambda d: d["Count"] / n_runs) \
              .sort_values("Frequency", ascending=True))
        plt.figure(figsize=(8, 0.3 * len(df)))
        ax = df["Frequency"].plot.barh()
        ax.set_title(f"Feature Selection Frequency (f={fsize}) \u2014 {n_runs} runs")
        ax.set_xlabel("Proportion of runs in which feature was selected")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"feature_frequency_f{fsize}.png")
        plt.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved frequency plot to: {out_file}")
        with open(LOG_FILE, "a") as f:
            f.write(f"  Saved frequency plot f{fsize}: {out_file}\n")


if __name__ == "__main__":
    print("\n--- 11. Generating Additional Visualizations ---")
    roc_plot = os.path.join(DIRS["plots_classification"], "roc_feature_grid.png")
    plot_mean_roc_feature_grid(roc_plot)

    boxplot_file = os.path.join(DIRS["plots_classification"], "auc_boxplot_models_features.png")
    plot_auc_boxplot(boxplot_file)

    feature_sets = load_best_feature_sets()
    upset_file = os.path.join(DIRS["plots_classification"], "upset_plot_features.png")
    heatmap_file = os.path.join(DIRS["plots_classification"], "heatmap_gene_presence.png")
    plot_upset(feature_sets, upset_file)
    plot_feature_heatmap(feature_sets, heatmap_file)

    freq_dir = os.path.join(DIRS["plots_classification"], "feature_frequency")
    plot_feature_frequencies(freq_dir)

    print("Additional visualizations complete.")
