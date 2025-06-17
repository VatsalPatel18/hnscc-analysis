# -*- coding: utf-8 -*-
"""
Utility script to generate Train/Test TCGA classification metric plots
using already-generated results (no changes to existing pipeline files).
Produces per-model line plots of mean scores and violin plots of run-level distributions.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to existing results and output directory for plots
RESULTS_CSV = os.path.join(
    "hnscc_robust_analysis_output", "results", "classification_metrics_detailed.csv"
)
OUT_DIR = os.path.join(
    "hnscc_robust_analysis_output", "plots", "classification_performance"
)

def main():
    # Load detailed metrics
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Metrics file not found: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)

    # Filter to TCGA Train/Test only
    df = df[df["Dataset"].isin(["Train (TCGA)", "Test (TCGA)"])]

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Line plots of mean scores
    agg = (
        df.groupby(["Model", "Dataset", "FeatureSize"])[metrics]
        .mean()
        .reset_index()
    )
    df_long = agg.melt(
        id_vars=["Model", "Dataset", "FeatureSize"],
        value_vars=metrics,
        var_name="Metric",
        value_name="MeanScore",
    )
    sns.set(style="whitegrid")
    for model in df_long["Model"].unique():
        sub = df_long[df_long["Model"] == model]
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=sub,
            x="FeatureSize",
            y="MeanScore",
            hue="Metric",
            style="Dataset",
            markers=True,
            dashes=False,
        )
        plt.title(f"{model}: Train/Test Mean Metrics by Feature Size")
        plt.ylim(0, 1)
        plt.xticks(sorted(sub["FeatureSize"].unique()))
        plt.xlabel("Number of Selected Features")
        plt.ylabel("Mean Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f"{model}_train_test_line.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved line plot for model {model}: {fname}")

    # 2) Violin plots of run-level distributions
    dfv = df.melt(
        id_vars=["Model", "Run", "Dataset", "FeatureSize"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    for model in dfv["Model"].unique():
        subv = dfv[dfv["Model"] == model]
        g = sns.catplot(
            data=subv,
            x="Metric",
            y="Score",
            hue="Dataset",
            col="FeatureSize",
            kind="violin",
            sharey=True,
            col_wrap=3,
            height=4,
        )
        g.fig.suptitle(f"{model}: Train/Test Metric Distributions by Feature Size", y=1.02)
        plt.ylim(0, 1)
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f"{model}_train_test_violin.png")
        g.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved violin plot for model {model}: {fname}")


if __name__ == "__main__":
    main()