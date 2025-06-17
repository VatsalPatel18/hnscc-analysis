# -*- coding: utf-8 -*-
"""Step 5: Binary classification loop over multiple runs.

This step now records train/test metrics, optional CPTAC metrics (if labels are
available), and saves per-sample prediction tables for all runs.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from step0_setup import (
    FEATURE_SIZES,
    RUNS,
    CLASSIFIERS,
    DIRS,
    LOG_FILE,
    CPTAC_EXPRESSION_FILE,
)
from step4_prepare_binary_classification import (
    X_full_binary,
    y_full_binary,
    feature_names_all,
    df_expr_binary,
    df_binary_subset,
)

def main():
    print(
        f"\n--- 5. Starting Binary Classification Loop ({RUNS} Runs, Feature Sizes: {FEATURE_SIZES}) ---"
    )
    classification_results_list = []
    detailed_metrics_list = []
    predictions_list = []
    roc_data_storage = {}

    try:
        df_expr_cptac = pd.read_csv(CPTAC_EXPRESSION_FILE, index_col="PatientID")
    except Exception as e:
        print(f"Warning: Could not load CPTAC expression data: {e}")
        df_expr_cptac = pd.DataFrame()

    for fsize in FEATURE_SIZES:
        print(f"\n=== Feature Selection Size: {fsize} ===")
        for model_name in CLASSIFIERS.keys():
            roc_data_storage[(model_name, fsize)] = []

        patient_ids = df_expr_binary.index.values
        for run in range(1, RUNS + 1):
            print(f"  --- Run {run}/{RUNS} ---")
            idx_train, idx_test = train_test_split(
                np.arange(len(y_full_binary)),
                test_size=0.3,
                random_state=42 + run,
                stratify=y_full_binary,
            )
            X_train = X_full_binary[idx_train]
            X_test = X_full_binary[idx_test]
            y_train = y_full_binary[idx_train]
            y_test = y_full_binary[idx_test]
            k_fs = min(fsize, X_train.shape[1])
            if k_fs < 1:
                print(f"Warning: Feature size {fsize} resulted in k={k_fs}. Skipping.")
                continue
            print(f"    Selecting top {k_fs} features...")
            selector = SelectKBest(score_func=f_classif, k=k_fs)
            try:
                X_train_fs = selector.fit_transform(X_train, y_train)
                X_test_fs = selector.transform(X_test)
            except ValueError as e:
                print(f"    Error during feature selection fsize={fsize}, run={run}: {e}")
                continue

            selected_indices = selector.get_support(indices=True)
            selected_features = feature_names_all[selected_indices]
            feature_file = os.path.join(
                DIRS["results_features"], f"selected_features_f{fsize}_run{run}.txt"
            )
            with open(feature_file, "w") as f_out:
                f_out.write("\n".join(selected_features) + "\n")

            for model_name, clf_prototype in CLASSIFIERS.items():
                print(f"      Training {model_name}...")
                clf = clf_prototype
                try:
                    clf.fit(X_train_fs, y_train)

                    # Predictions on training data
                    y_train_pred = clf.predict(X_train_fs)
                    y_train_proba = clf.predict_proba(X_train_fs)[:, 1]
                    acc_tr = accuracy_score(y_train, y_train_pred)
                    prec_tr = precision_score(y_train, y_train_pred, zero_division=0)
                    rec_tr = recall_score(y_train, y_train_pred, zero_division=0)
                    f1_tr = f1_score(y_train, y_train_pred, zero_division=0)
                    auc_tr = roc_auc_score(y_train, y_train_proba)

                    # Predictions on test data
                    y_pred = clf.predict(X_test_fs)
                    y_proba = clf.predict_proba(X_test_fs)[:, 1]
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    auc = roc_auc_score(y_test, y_proba)
                    fpr, tpr, _ = roc_curve(y_test, y_proba)

                    # Predictions on CPTAC if possible
                    acc_cpt = prec_cpt = rec_cpt = f1_cpt = auc_cpt = np.nan
                    cptac_ids = []
                    cptac_proba = []
                    if not df_expr_cptac.empty:
                        try:
                            X_cptac_fs = df_expr_cptac[selected_features].values
                            cptac_proba = clf.predict_proba(X_cptac_fs)[:, 1]
                            # If CPTAC has true labels use them, else keep NaN metrics
                            if "Label" in df_expr_cptac.columns:
                                y_cpt = df_expr_cptac["Label"].values
                                y_cpt_pred = clf.predict(X_cptac_fs)
                                acc_cpt = accuracy_score(y_cpt, y_cpt_pred)
                                prec_cpt = precision_score(y_cpt, y_cpt_pred, zero_division=0)
                                rec_cpt = recall_score(y_cpt, y_cpt_pred, zero_division=0)
                                f1_cpt = f1_score(y_cpt, y_cpt_pred, zero_division=0)
                                auc_cpt = roc_auc_score(y_cpt, cptac_proba)
                        except Exception as e:
                            print(f"        Warning: CPTAC prediction failed: {e}")

                    classification_results_list.append(
                        {
                            "FeatureSize": fsize,
                            "Run": run,
                            "Model": model_name,
                            "Accuracy": acc,
                            "Precision": prec,
                            "Recall": rec,
                            "F1": f1,
                            "AUC": auc,
                            "NumSelectedFeatures": k_fs,
                        }
                    )
                    # store detailed metrics for each dataset
                    detailed_metrics_list.extend([
                        {
                            "FeatureSize": fsize,
                            "Run": run,
                            "Model": model_name,
                            "Dataset": "Train (TCGA)",
                            "Accuracy": acc_tr,
                            "Precision": prec_tr,
                            "Recall": rec_tr,
                            "F1": f1_tr,
                            "AUC": auc_tr,
                            "NumSelectedFeatures": k_fs,
                        },
                        {
                            "FeatureSize": fsize,
                            "Run": run,
                            "Model": model_name,
                            "Dataset": "Test (TCGA)",
                            "Accuracy": acc,
                            "Precision": prec,
                            "Recall": rec,
                            "F1": f1,
                            "AUC": auc,
                            "NumSelectedFeatures": k_fs,
                        },
                        {
                            "FeatureSize": fsize,
                            "Run": run,
                            "Model": model_name,
                            "Dataset": "CPTAC",
                            "Accuracy": acc_cpt,
                            "Precision": prec_cpt,
                            "Recall": rec_cpt,
                            "F1": f1_cpt,
                            "AUC": auc_cpt,
                            "NumSelectedFeatures": k_fs,
                        },
                    ])

                    # store prediction details
                    for pid, tlabel, plab, pprob in zip(
                        patient_ids[idx_train],
                        y_train,
                        y_train_pred,
                        y_train_proba,
                    ):
                        predictions_list.append(
                            {
                                "FeatureSize": fsize,
                                "Run": run,
                                "Model": model_name,
                                "Dataset": "Train (TCGA)",
                                "PatientID": pid,
                                "TrueLabel": tlabel,
                                "PredictedLabel": plab,
                                "PredictedProbability_Class1": pprob,
                            }
                        )

                    for pid, tlabel, plab, pprob in zip(
                        patient_ids[idx_test],
                        y_test,
                        y_pred,
                        y_proba,
                    ):
                        predictions_list.append(
                            {
                                "FeatureSize": fsize,
                                "Run": run,
                                "Model": model_name,
                                "Dataset": "Test (TCGA)",
                                "PatientID": pid,
                                "TrueLabel": tlabel,
                                "PredictedLabel": plab,
                                "PredictedProbability_Class1": pprob,
                            }
                        )

                    if not df_expr_cptac.empty:
                        cptac_ids = df_expr_cptac.index.tolist()
                        for pid, pprob in zip(cptac_ids, cptac_proba):
                            predictions_list.append(
                                {
                                    "FeatureSize": fsize,
                                    "Run": run,
                                    "Model": model_name,
                                    "Dataset": "CPTAC",
                                    "PatientID": pid,
                                    "TrueLabel": df_expr_cptac["Label"].loc[pid]
                                    if "Label" in df_expr_cptac.columns
                                    else np.nan,
                                    "PredictedLabel": np.nan
                                    if "Label" not in df_expr_cptac.columns
                                    else clf.predict(X_cptac_fs)[
                                        list(df_expr_cptac.index).index(pid)
                                    ],
                                    "PredictedProbability_Class1": pprob,
                                }
                            )
                    roc_data_storage[(model_name, fsize)].append(
                        {"run": run, "fpr": fpr, "tpr": tpr, "auc": auc}
                    )

                    model_filename = os.path.join(
                        DIRS["models_classification"], f"{model_name}_f{fsize}_run{run}.pkl"
                    )
                    with open(model_filename, "wb") as f_out:
                        pickle.dump(
                            {"model": clf, "selector": selector, "features": selected_features},
                            f_out,
                        )
                except Exception as e:
                    print(
                        f"      Error training/predicting {model_name} fsize={fsize}, run={run}: {e}"
                    )
                    classification_results_list.append(
                        {
                            "FeatureSize": fsize,
                            "Run": run,
                            "Model": model_name,
                            "Accuracy": np.nan,
                            "Precision": np.nan,
                            "Recall": np.nan,
                            "F1": np.nan,
                            "AUC": np.nan,
                            "NumSelectedFeatures": k_fs,
                        }
                    )
                    roc_data_storage[(model_name, fsize)].append(
                        {
                            "run": run,
                            "fpr": np.array([0, 1]),
                            "tpr": np.array([0, 1]),
                            "auc": np.nan,
                        }
                    )
                    with open(LOG_FILE, "a") as f:
                        f.write(
                            f"Error training/predicting: model={model_name}, fsize={fsize}, run={run}, error={e}\n"
                        )

    results_df = pd.DataFrame(classification_results_list)
    results_csv = os.path.join(DIRS["results"], "classification_metrics_all_runs.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved detailed classification metrics to: {results_csv}")

    detailed_df = pd.DataFrame(detailed_metrics_list)
    detailed_csv = os.path.join(DIRS["results"], "classification_metrics_detailed.csv")
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"Saved extended metrics to: {detailed_csv}")

    pred_df = pd.DataFrame(predictions_list)
    pred_csv = os.path.join(DIRS["results_predictions"], "all_predictions_all_runs.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved all prediction outputs to: {pred_csv}")

    roc_data_file = os.path.join(DIRS["results"], "roc_data_all_runs.pkl")
    with open(roc_data_file, "wb") as f:
        pickle.dump(roc_data_storage, f)
    print(f"Saved ROC curve data to: {roc_data_file}")

    with open(LOG_FILE, "a") as f:
        f.write("\nBinary Classification Loop Completed:\n")
        f.write(f"  Saved detailed metrics: {results_csv}\n")
        f.write(f"  Saved extended metrics: {detailed_csv}\n")
        f.write(f"  Saved prediction table: {pred_csv}\n")
        f.write(f"  Saved ROC data: {roc_data_file}\n")
        f.write(f"  Saved models/selectors in: {DIRS['models_classification']}\n")
        f.write(f"  Saved selected features in: {DIRS['results_features']}\n")

    print("\nBinary classification loop complete.")
    return results_df, roc_data_storage


if __name__ == "__main__":
    main()
