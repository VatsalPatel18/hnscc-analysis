# -*- coding: utf-8 -*-
"""Step 8: Select overall best model based on combined criteria.

In addition to writing the best configuration to disk, this script exposes
``selected_features_final_combined``
containing the feature list from that configuration.  Step 10 imports this
variable when re-clustering using only those genes.
"""

import os
import pandas as pd

from step0_setup import (
    DIRS,
    P_VALUE_THRESHOLD_TCGA,
    P_VALUE_THRESHOLD_CPTAC,
    P_VALUE_THRESHOLD_CPTAC_RELAXED,
    LOG_FILE,
)
from step7_validation_summary import summary_df

print("\n--- 8. Selecting Overall Best Model (Combined Criteria) ---")

if summary_df is None or summary_df.empty:
    raise RuntimeError("summary_df is empty. Ensure step7_validation_summary completed successfully.")

df = summary_df.copy()
df["TCGA_Significant"] = df["FullTCGA_LogRank_PValue"] < P_VALUE_THRESHOLD_TCGA
df["CPTAC_Significant_Strict"] = df["CPTAC_LogRank_PValue"] < P_VALUE_THRESHOLD_CPTAC
df["CPTAC_Significant_Relaxed"] = df["CPTAC_LogRank_PValue"] < P_VALUE_THRESHOLD_CPTAC_RELAXED

# First try strict CPTAC threshold
candidates = df[df["TCGA_Significant"] & df["CPTAC_Significant_Strict"]].dropna(subset=["MeanTestAUC_Subset"])
criteria_used = "strict"

# If none satisfy strict criteria, fall back to relaxed CPTAC threshold
if candidates.empty:
    candidates = df[df["TCGA_Significant"] & df["CPTAC_Significant_Relaxed"]].dropna(subset=["MeanTestAUC_Subset"])
    criteria_used = "relaxed"

# If still none, simply choose the configuration with best mean AUC
if candidates.empty:
    print("Warning: No configuration met significance criteria. Selecting by mean AUC only.")
    candidates = df.dropna(subset=["MeanTestAUC_Subset"])
    criteria_used = "auc_only"

if candidates.empty:
    raise RuntimeError("No valid configurations available for selection.")

best_row = candidates.loc[candidates["MeanTestAUC_Subset"].idxmax()]
overall_best_config = best_row.to_dict()

best_config_file = os.path.join(DIRS["results"], "overall_best_model.csv")
pd.DataFrame([best_row]).to_csv(best_config_file, index=False, float_format="%.4g")

print("\n--- Overall Best Configuration ---")
print(best_row.to_string())
print(f"\nSaved best configuration to: {best_config_file}")

with open(LOG_FILE, "a") as f:
    f.write("\nOverall Best Model Selection (Step 8):\n")
    f.write(f"  Criteria used: {criteria_used}\n")
    f.write(
        f"  Model: {overall_best_config['BestModel']}, FeatureSize: {overall_best_config['FeatureSize']}, "
        f"Mean AUC: {overall_best_config['MeanTestAUC_Subset']:.4g}\n"
    )
    f.write(
        f"  TCGA p: {overall_best_config['FullTCGA_LogRank_PValue']:.4g}, CPTAC p: {overall_best_config['CPTAC_LogRank_PValue']:.4g}\n"
    )
    f.write(f"  Saved selection: {best_config_file}\n")

print("\nOverall best model selection complete.")

# ---------------------------------------------------------------------------
# Load feature list for the overall best configuration
# ---------------------------------------------------------------------------
# This list is imported by step10_reclustering.py to perform clustering on
# TCGA expression data using only the genes selected for the best performing
# classification model.
selected_features_final_combined = []
try:
    best_run = int(overall_best_config.get("BestRun"))
    fsize = int(overall_best_config["FeatureSize"])
    feature_file = os.path.join(
        DIRS["results_features"], f"selected_features_f{fsize}_run{best_run}.txt"
    )
    if os.path.exists(feature_file):
        with open(feature_file) as f:
            selected_features_final_combined = [
                line.strip() for line in f if line.strip()
            ]
        print(
            f"Loaded {len(selected_features_final_combined)} features from {feature_file}"
        )
    else:
        print(f"Feature file not found: {feature_file}")
except Exception as e:
    print(f"Error loading selected features for final configuration: {e}")

