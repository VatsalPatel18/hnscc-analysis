# -*- coding: utf-8 -*-
"""Step 7: Validate models and create summary tables."""

import os
import pickle
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

from step0_setup import DIRS, CPTAC_EXPRESSION_FILE, CPTAC_SURVIVAL_FILE, P_VALUE_THRESHOLD_TCGA, P_VALUE_THRESHOLD_CPTAC, P_VALUE_THRESHOLD_CPTAC_RELAXED, LOG_FILE
from step1_load_tcga import df_expr_tcga, df_surv_tcga, required_survival_cols

results_csv = os.path.join(DIRS["results"], "classification_metrics_all_runs.csv")
agg_results_csv = os.path.join(DIRS["results"], "classification_metrics_aggregated.csv")

if not os.path.exists(results_csv) or not os.path.exists(agg_results_csv):
    raise RuntimeError(
        "Required result files not found. Run previous steps before validation."
    )

results_df = pd.read_csv(results_csv)
df_agg = pd.read_csv(agg_results_csv)

# Directory for raw prediction outputs (outside project)
DOWNLOAD_PRED_DIR = os.path.expanduser(os.path.join('~', 'Downloads', 'new_folder'))
os.makedirs(DOWNLOAD_PRED_DIR, exist_ok=True)

print("\n--- 7. Starting Validation on Full Datasets and Summary Table Generation ---")

df_expr_cptac = None
df_surv_cptac = None
try:
    df_expr_cptac = pd.read_csv(CPTAC_EXPRESSION_FILE, index_col="PatientID")
    df_surv_cptac = pd.read_csv(CPTAC_SURVIVAL_FILE, index_col="PatientID")
    print(f"\nCPTAC Data Shapes: Expr={df_expr_cptac.shape}, Surv={df_surv_cptac.shape}")
    if not all(col in df_surv_cptac.columns for col in required_survival_cols):
        raise ValueError(f"CPTAC Survival data missing required columns: {required_survival_cols}")
    common_patients_cptac = df_expr_cptac.index.intersection(df_surv_cptac.index)
    df_expr_cptac = df_expr_cptac.loc[common_patients_cptac]
    df_surv_cptac = df_surv_cptac.loc[common_patients_cptac]
    print(f"Aligned CPTAC shapes: Expr={df_expr_cptac.shape}, Surv={df_surv_cptac.shape}")
    with open(LOG_FILE, "a") as f:
        f.write(f"\nCPTAC Data Loaded & Aligned: Expr={df_expr_cptac.shape}, Surv={df_surv_cptac.shape}\n")
except Exception as e:
    print(f"Error loading or processing CPTAC data: {e}. Skipping CPTAC validation.")
    with open(LOG_FILE, "a") as f:
        f.write(f"\nError loading/processing CPTAC data: {e}. Skipping CPTAC validation.\n")
    df_expr_cptac = None
    df_surv_cptac = None

common_features = None
if df_expr_cptac is not None:
    common_features = df_expr_tcga.columns.intersection(df_expr_cptac.columns)
    print(f"\nFound {len(common_features)} common features between TCGA and CPTAC.")
    if len(common_features) == 0:
        print("Warning: No common features found. Cannot perform CPTAC validation.")
        with open(LOG_FILE, "a") as f:
            f.write("\nWarning: No common features between TCGA/CPTAC. Skipping CPTAC validation.\n")
        df_expr_cptac = None
        df_surv_cptac = None
    else:
        df_expr_cptac = df_expr_cptac[common_features]
        print(f"CPTAC expression data subsetted to {df_expr_cptac.shape[1]} common features.")

summary_data = []
for fsize in df_agg["FeatureSize"].unique():
    print(f"\n--- Processing Feature Size: {fsize} ---")
    df_fsize_agg = df_agg[df_agg["FeatureSize"] == fsize]
    if df_fsize_agg.empty:
        print(f"  No aggregated results for fsize={fsize}. Skipping.")
        continue
    df_fsize_agg_valid = df_fsize_agg.dropna(subset=['mean'])
    if df_fsize_agg_valid.empty:
        print(f"  No valid mean AUC for fsize={fsize}. Skipping.")
        continue
    best_model_row_fsize = df_fsize_agg_valid.loc[df_fsize_agg_valid["mean"].idxmax()]
    best_model_fsize = best_model_row_fsize["Model"]
    mean_auc_fsize = best_model_row_fsize["mean"]
    print(f"  Best model for f={fsize}: {best_model_fsize} (Mean AUC = {mean_auc_fsize:.4f})")
    df_runs = results_df[(results_df["FeatureSize"] == fsize) & (results_df["Model"] == best_model_fsize)].dropna(subset=['AUC'])
    if df_runs.empty:
        print(f"  No valid runs for {best_model_fsize} f={fsize}. Skipping.")
        continue
    best_run_row = df_runs.loc[df_runs["AUC"].idxmax()]
    best_run_fsize = int(best_run_row["Run"])
    print(f"  Best run: {best_run_fsize} (AUC = {best_run_row['AUC']:.4f})")

    model_path = os.path.join(DIRS["models_classification"], f"{best_model_fsize}_f{fsize}_run{best_run_fsize}.pkl")
    try:
        with open(model_path, "rb") as f:
            loaded_data = pickle.load(f)
        clf_fsize = loaded_data["model"]
        selected_features_fsize = loaded_data["features"]
        print(f"  Loaded model and {len(selected_features_fsize)} features from: {model_path}")
    except Exception as e:
        print(f"  Error loading model {model_path}: {e}. Skipping.")
        continue

    tcga_p_value = np.nan
    print("  Predicting on full TCGA...")
    try:
        if not isinstance(selected_features_fsize, (list, np.ndarray, pd.Index)):
            raise TypeError(f"selected_features_fsize is not list-like: {type(selected_features_fsize)}")
        missing_tcga_feats = set(list(selected_features_fsize)) - set(df_expr_tcga.columns)
        if missing_tcga_feats:
            raise ValueError(f"Features missing in full TCGA: {missing_tcga_feats}")
        X_tcga_full_fs = df_expr_tcga[selected_features_fsize].values
        tcga_pred_labels = clf_fsize.predict(X_tcga_full_fs)
        tcga_pred_probas = clf_fsize.predict_proba(X_tcga_full_fs)[:, 1]
        df_tcga_preds = pd.DataFrame({'PredictedLabel': tcga_pred_labels, 'PredictedProbability_Class1': tcga_pred_probas}, index=df_expr_tcga.index)
        pred_file_tcga = os.path.join(DIRS["results_predictions"], f"tcga_full_predictions_f{fsize}_model_{best_model_fsize}_run{best_run_fsize}.csv")
        df_tcga_preds.to_csv(pred_file_tcga)
        # Also save raw TCGA predictions to user downloads folder
        raw_file_tcga = os.path.join(DOWNLOAD_PRED_DIR, os.path.basename(pred_file_tcga))
        df_tcga_preds.to_csv(raw_file_tcga)
        print(f"Saved raw TCGA predictions to: {raw_file_tcga}")
        with open(LOG_FILE, "a") as f:
            f.write(f"  Raw TCGA predictions saved to: {raw_file_tcga}\n")

        df_surv_tcga_pred = df_surv_tcga.loc[df_expr_tcga.index].copy()
        df_surv_tcga_pred["Classifier_Group"] = tcga_pred_labels
        group0_tcga = df_surv_tcga_pred[df_surv_tcga_pred["Classifier_Group"] == 0]
        group1_tcga = df_surv_tcga_pred[df_surv_tcga_pred["Classifier_Group"] == 1]
        if not group0_tcga.empty and not group1_tcga.empty and group0_tcga["Overall Survival Status"].sum() > 0 and group1_tcga["Overall Survival Status"].sum() > 0:
            result_tcga = logrank_test(group0_tcga["Overall Survival (Months)"], group1_tcga["Overall Survival (Months)"], event_observed_A=group0_tcga["Overall Survival Status"], event_observed_B=group1_tcga["Overall Survival Status"])
            tcga_p_value = result_tcga.p_value
            print(f"  Full TCGA Log-rank p-value: {tcga_p_value:.4g}")
        else:
            print("  Warning: Cannot calculate TCGA log-rank (empty group or no events).")
    except Exception as e:
        print(f"  Error during full TCGA validation f={fsize}: {e}")
        with open(LOG_FILE, "a") as f:
            f.write(f"Error TCGA validation f={fsize}: {e}\n")

    cptac_p_value = np.nan
    if df_expr_cptac is not None:
        print("  Predicting on CPTAC...")
        try:
            if not isinstance(selected_features_fsize, (list, np.ndarray, pd.Index)):
                raise TypeError(f"selected_features_fsize is not list-like: {type(selected_features_fsize)}")
            required_common_features = list(set(list(selected_features_fsize)).intersection(common_features))
            if len(required_common_features) != len(selected_features_fsize):
                print(f"  Warning: Only {len(required_common_features)}/{len(selected_features_fsize)} model features are common with CPTAC.")
            if not required_common_features:
                raise ValueError("No common features for CPTAC prediction.")
            X_cptac_fs = df_expr_cptac[required_common_features].values
            cptac_pred_labels = clf_fsize.predict(X_cptac_fs)
            cptac_pred_probas = clf_fsize.predict_proba(X_cptac_fs)[:, 1]
            df_cptac_preds = pd.DataFrame({'PredictedLabel': cptac_pred_labels, 'PredictedProbability_Class1': cptac_pred_probas}, index=df_expr_cptac.index)
            pred_file_cptac = os.path.join(DIRS["results_predictions"], f"cptac_predictions_f{fsize}_model_{best_model_fsize}_run{best_run_fsize}.csv")
            df_cptac_preds.to_csv(pred_file_cptac)
            # Also save raw CPTAC predictions to user downloads folder
            raw_file_cptac = os.path.join(DOWNLOAD_PRED_DIR, os.path.basename(pred_file_cptac))
            df_cptac_preds.to_csv(raw_file_cptac)
            print(f"Saved raw CPTAC predictions to: {raw_file_cptac}")
            with open(LOG_FILE, "a") as f:
                f.write(f"  Raw CPTAC predictions saved to: {raw_file_cptac}\n")

            df_surv_cptac_pred = df_surv_cptac.loc[df_expr_cptac.index].copy()
            df_surv_cptac_pred["Classifier_Group"] = cptac_pred_labels
            group0_cptac = df_surv_cptac_pred[df_surv_cptac_pred["Classifier_Group"] == 0]
            group1_cptac = df_surv_cptac_pred[df_surv_cptac_pred["Classifier_Group"] == 1]
            if not group0_cptac.empty and not group1_cptac.empty and group0_cptac["Overall Survival Status"].sum() > 0 and group1_cptac["Overall Survival Status"].sum() > 0:
                result_cptac = logrank_test(group0_cptac["Overall Survival (Months)"], group1_cptac["Overall Survival (Months)"], event_observed_A=group0_cptac["Overall Survival Status"], event_observed_B=group1_cptac["Overall Survival Status"])
                cptac_p_value = result_cptac.p_value
                print(f"  CPTAC Log-rank p-value: {cptac_p_value:.4g}")
            else:
                print("  Warning: Cannot calculate CPTAC log-rank (empty group or no events).")
        except Exception as e:
            print(f"  Error during CPTAC validation f={fsize}: {e}")
            with open(LOG_FILE, "a") as f:
                f.write(f"Error CPTAC validation f={fsize}: {e}\n")

    summary_data.append({
        "FeatureSize": fsize,
        "BestModel": best_model_fsize,
        "MeanTestAUC_Subset": mean_auc_fsize,
        "BestRun": best_run_fsize,
        "FullTCGA_LogRank_PValue": tcga_p_value,
        "CPTAC_LogRank_PValue": cptac_p_value,
    })

summary_df = pd.DataFrame(summary_data)
summary_table_file = os.path.join(DIRS["results"], "performance_summary_table.csv")
summary_df.to_csv(summary_table_file, index=False, float_format='%.4g')
print(f"\nSaved performance summary table to: {summary_table_file}")
print("\n--- Performance Summary Table ---")
print(summary_df.to_string())
with open(LOG_FILE, "a") as f:
    f.write("\nValidation and Summary Generation Completed:\n")
    f.write(f"  Saved prediction files in: {DIRS['results_predictions']}\n")
    f.write(f"  Saved summary table: {summary_table_file}\n")
    f.write("\nSummary Table:\n" + summary_df.to_string() + "\n")

print("\nValidation and summary table generation complete.")
