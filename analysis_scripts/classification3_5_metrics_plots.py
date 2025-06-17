# -*- coding: utf-8 -*-
"""Step classification3.5: Visualization of train/test metrics across datasets."""

import os
import pandas as pd
import matplotlib.pyplot as plt

from step0_setup import DIRS, LOG_FILE

metrics_file = os.path.join(DIRS["results"], "classification_metrics_detailed.csv")
if not os.path.exists(metrics_file):
    raise RuntimeError(
        "Detailed metrics file not found. Run step5_classification_loop first."
    )

df = pd.read_csv(metrics_file)

print("\n--- classification3.5: Plotting average metrics across datasets ---")

metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
agg = (
    df.groupby("Dataset")[metric_cols]
    .mean()
    .reindex(["Train (TCGA)", "Test (TCGA)", "CPTAC"])
)

fig, ax = plt.subplots(figsize=(8, 6))
agg.plot(kind="bar", ax=ax)
ax.set_title("Average Classification Metrics by Dataset")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_file = os.path.join(
    DIRS["plots_classification"], "avg_metrics_by_dataset.png"
)
plt.savefig(out_file, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved metric comparison plot to: {out_file}")

print("Generating per-feature-size and per-model plots...")

agg_fm = (
    df.groupby(["FeatureSize", "Model", "Dataset"])[metric_cols]
    .mean()
    .reset_index()
)

feature_sizes = sorted(df["FeatureSize"].unique())
models = sorted(df["Model"].unique())

for fsize in feature_sizes:
    for model in models:
        sub = agg_fm[(agg_fm["FeatureSize"] == fsize) & (agg_fm["Model"] == model)]
        if sub.empty:
            continue
        sub = sub.set_index("Dataset")[metric_cols].reindex(
            ["Train (TCGA)", "Test (TCGA)", "CPTAC"]
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sub.plot(kind="bar", ax=ax)
        ax.set_title(f"{model} (f={fsize}) Avg Metrics by Dataset")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(
            DIRS["plots_classification"], f"avg_metrics_{model}_f{fsize}.png"
        )
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved: {fname}")

with open(LOG_FILE, "a") as f:
    f.write("\nclassification3.5 Visualization:\n")
    f.write(f"  Saved metrics plot: {out_file}\n")
    for fsize in feature_sizes:
        for model in models:
            plot_path = os.path.join(
                DIRS["plots_classification"], f"avg_metrics_{model}_f{fsize}.png"
            )
            if os.path.exists(plot_path):
                f.write(f"  Saved metrics plot: {plot_path}\n")

print("Metric visualization complete.")
