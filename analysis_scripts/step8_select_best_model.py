# -*- coding: utf-8 -*-
"""Step 8: Select overall best model based on combined criteria."""

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

