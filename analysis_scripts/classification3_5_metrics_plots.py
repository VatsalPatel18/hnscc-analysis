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

metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
agg = df.groupby("Dataset")[metric_cols].mean().reindex([
    "Train (TCGA)",
    "Test (TCGA)",
    "CPTAC",
])

fig, ax = plt.subplots(figsize=(8, 6))
agg.plot(kind="bar", ax=ax)
ax.set_title("Average Classification Metrics by Dataset")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_file = os.path.join(DIRS["plots_classification"], "avg_metrics_by_dataset.png")
plt.savefig(out_file, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved metric comparison plot to: {out_file}")

with open(LOG_FILE, "a") as f:
    f.write("\nclassification3.5 Visualization:\n")
    f.write(f"  Saved metrics plot: {out_file}\n")

print("Metric visualization complete.")
