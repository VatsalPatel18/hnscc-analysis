# -*- coding: utf-8 -*-
"""Step 12: Line and violin plots of Train/Test TCGA metrics across feature sizes for each model."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from step0_setup import DIRS, FEATURE_SIZES, CLASSIFIERS, LOG_FILE

METRICS_CSV = os.path.join(DIRS["results"], "classification_metrics_detailed.csv")
METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
DATASETS = ["Train (TCGA)", "Test (TCGA)"]


def plot_line_metrics(df, model_name):
    """Line plot of mean metrics vs. feature size, separated by dataset and metric."""
    dfm = df[(df.Model == model_name) & df.Dataset.isin(DATASETS)]
    if dfm.empty:
        return
    agg = (
        dfm.groupby(["FeatureSize", "Dataset"] + [])
           [METRIC_COLS]
           .mean()
           .reset_index()
    )
    df_long = agg.melt(
        id_vars=["FeatureSize", "Dataset"], value_vars=METRIC_COLS,
        var_name="Metric", value_name="MeanScore"
    )
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_long, x="FeatureSize", y="MeanScore",
        hue="Metric", style="Dataset", markers=True, dashes=False
    )
    plt.title(f"{model_name}: Train/Test Average Metrics by Feature Size")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Mean Score")
    plt.xticks(FEATURE_SIZES)
    plt.ylim(0, 1)
    plt.legend(title="Metric / Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_file = os.path.join(
        DIRS["plots_classification"], f"{model_name}_line_metrics_train_test.png"
    )
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved line plot: {out_file}")
    with open(LOG_FILE, "a") as log:
        log.write(f"Saved line plot: {out_file}\n")


def plot_violin_metrics(df, model_name):
    """Violin plots of metrics distributions by feature size and dataset."""
    dfm = df[(df.Model == model_name) & df.Dataset.isin(DATASETS)]
    if dfm.empty:
        return
    df_long = dfm.melt(
        id_vars=["FeatureSize", "Run", "Dataset"], value_vars=METRIC_COLS,
        var_name="Metric", value_name="Score"
    )
    sns.set(style="whitegrid")
    g = sns.catplot(
        data=df_long, x="Metric", y="Score", hue="Dataset", col="FeatureSize",
        kind="violin", sharey=True, col_wrap=3, height=4
    )
    g.set_titles("f={col_name}")
    g.set_axis_labels("Metric", "Score")
    plt.ylim(0, 1)
    g.fig.suptitle(f"{model_name}: Train/Test Metric Distributions by Feature Size", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(
        DIRS["plots_classification"], f"{model_name}_violin_metrics_train_test.png"
    )
    g.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved violin plot: {out_file}")
    with open(LOG_FILE, "a") as log:
        log.write(f"Saved violin plot: {out_file}\n")


if __name__ == "__main__":
    print("\n--- 12. Generating Train/Test TCGA Metric Plots by Model ---")
    if not os.path.exists(METRICS_CSV):
        print(f"Metrics CSV not found: {METRICS_CSV}")
    else:
        df = pd.read_csv(METRICS_CSV)
        for model in CLASSIFIERS.keys():
            plot_line_metrics(df, model)
            plot_violin_metrics(df, model)
    print("Completed Train/Test metric visualizations.")