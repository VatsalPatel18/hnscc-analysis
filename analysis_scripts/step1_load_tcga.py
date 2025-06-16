# -*- coding: utf-8 -*-
"""Step 1: Load TCGA data for HNSCC analysis."""

import pandas as pd
from step0_setup import TCGA_EXPRESSION_FILE, TCGA_SURVIVAL_FILE, LOG_FILE

print("\n--- 1. Loading TCGA Data ---")
try:
    df_expr_tcga = pd.read_csv(TCGA_EXPRESSION_FILE, index_col="PatientID")
    df_surv_tcga = pd.read_csv(TCGA_SURVIVAL_FILE, index_col="PatientID")
except FileNotFoundError as e:
    print(f"Error loading TCGA data: {e}")
    raise

print(f"Initial TCGA Expression Shape: {df_expr_tcga.shape}")
print(f"Initial TCGA Survival Shape:   {df_surv_tcga.shape}")

required_survival_cols = ["Overall Survival (Months)", "Overall Survival Status"]
if not all(col in df_surv_tcga.columns for col in required_survival_cols):
    raise ValueError(f"Missing required survival columns in TCGA data: {required_survival_cols}")

common_patients = df_expr_tcga.index.intersection(df_surv_tcga.index)
print(f"Number of patients common to TCGA expression and survival: {len(common_patients)}")
if len(common_patients) < df_expr_tcga.shape[0] or len(common_patients) < df_surv_tcga.shape[0]:
    print("Warning: Subsetting TCGA data to common patients.")
    df_expr_tcga = df_expr_tcga.loc[common_patients]
    df_surv_tcga = df_surv_tcga.loc[common_patients]

print(f"Aligned TCGA Expression Shape: {df_expr_tcga.shape}")
print(f"Aligned TCGA Survival Shape:   {df_surv_tcga.shape}")

with open(LOG_FILE, "a") as f:
    f.write(f"\nTCGA Data Loaded & Aligned:\n")
    f.write(f"  Expression: {df_expr_tcga.shape[0]} patients, {df_expr_tcga.shape[1]} features\n")
    f.write(f"  Survival: {df_surv_tcga.shape[0]} patients\n")

print("TCGA data loaded successfully.")
