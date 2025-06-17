# -*- coding: utf-8 -*-
"""Step 9: Generate final Kaplan-Meier plots using the best classifier.

This script loads ``overall_best_model.pkl`` produced by step8_select_best_model
and applies the saved model and feature list to the TCGA dataset (and CPTAC if
available) to create final survival curves.
"""

import os
import pickle

import pandas as pd

from step0_setup import DIRS, LOG_FILE
from step1_load_tcga import df_expr_tcga, df_surv_tcga
from step2_helper_functions import run_multi_group_survival

try:  # CPTAC data are optional
    from step7_validation_summary import df_expr_cptac, df_surv_cptac, common_features
except Exception:  # pragma: no cover - used when step7 was not executed
    df_expr_cptac = None
    df_surv_cptac = None
    common_features = None

print("\n--- 9. Final KM Plots (Overall Best Model) ---")

best_model_file = os.path.join(DIRS["models_classification"], "overall_best_model.pkl")

if not os.path.exists(best_model_file):
    raise RuntimeError(
        f"Overall best model file not found at {best_model_file}. Run step8_select_best_model first."
    )

try:
    with open(best_model_file, "rb") as f:
        best_model_data = pickle.load(f)
    clf = best_model_data["model"]
    selected_features = list(best_model_data["features"])
    model_name = best_model_data.get("model_name", "Model")
    feature_size = best_model_data.get("feature_size", len(selected_features))
    run_id = best_model_data.get("run", "?")
    print(
        f"Loaded overall best model: {model_name}, fsize={feature_size}, run={run_id}"
    )
except Exception as e:  # pragma: no cover - runtime error path
    raise RuntimeError(f"Could not load best model info from {best_model_file}: {e}")

print("Predicting TCGA labels...")
missing_feats_tcga = set(selected_features) - set(df_expr_tcga.columns)
if missing_feats_tcga:
    raise ValueError(f"Selected features missing in TCGA data: {missing_feats_tcga}")

X_tcga = df_expr_tcga[selected_features].values
tcga_labels = clf.predict(X_tcga)
df_surv_tcga_final = df_surv_tcga.loc[df_expr_tcga.index].copy()
df_surv_tcga_final["Classifier_Group"] = tcga_labels

print("Generating TCGA KM plot...")
prefix_tcga = os.path.join(
    DIRS["plots_validation"], f"tcga_final_{model_name}_f{feature_size}"
)
run_multi_group_survival(
    df_surv_tcga_final,
    cluster_col="Classifier_Group",
    plot_title=f"TCGA Survival by Final Classifier ({model_name}, f={feature_size})",
    out_prefix=prefix_tcga,
    show_plot=True,
)

plot_paths = [f"{prefix_tcga}_km_plot.png"]

if (
    df_expr_cptac is not None
    and df_surv_cptac is not None
    and common_features is not None
):
    print("Predicting CPTAC labels...")
    shared_feats = [f for f in selected_features if f in common_features]
    if shared_feats:
        X_cptac = df_expr_cptac[shared_feats].values
        cptac_labels = clf.predict(X_cptac)
        df_surv_cptac_final = df_surv_cptac.loc[df_expr_cptac.index].copy()
        df_surv_cptac_final["Classifier_Group"] = cptac_labels

        print("Generating CPTAC KM plot...")
        prefix_cptac = os.path.join(
            DIRS["plots_validation"], f"cptac_final_{model_name}_f{feature_size}"
        )
        run_multi_group_survival(
            df_surv_cptac_final,
            cluster_col="Classifier_Group",
            plot_title=f"CPTAC Survival by Final Classifier ({model_name}, f={feature_size})",
            out_prefix=prefix_cptac,
            show_plot=True,
        )
        plot_paths.append(f"{prefix_cptac}_km_plot.png")
    else:
        print("Selected features not present in CPTAC data - skipping CPTAC KM plot.")

with open(LOG_FILE, "a") as f:
    f.write("\nFinal KM Plots Generated:\n")
    for p in plot_paths:
        f.write(f"  {p}\n")

print("\nFinal KM plot generation complete.")

