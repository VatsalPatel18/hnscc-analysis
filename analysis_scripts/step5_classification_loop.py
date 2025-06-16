# -*- coding: utf-8 -*-
"""Step 5: Binary classification loop over multiple runs."""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from step0_setup import FEATURE_SIZES, RUNS, CLASSIFIERS, DIRS, LOG_FILE
from step4_prepare_binary_classification import X_full_binary, y_full_binary, feature_names_all

print(f"\n--- 5. Starting Binary Classification Loop ({RUNS} Runs, Feature Sizes: {FEATURE_SIZES}) ---")
classification_results_list = []
roc_data_storage = {}
for fsize in FEATURE_SIZES:
    print(f"\n=== Feature Selection Size: {fsize} ===")
    for model_name in CLASSIFIERS.keys():
        roc_data_storage[(model_name, fsize)] = []

    for run in range(1, RUNS + 1):
        print(f"  --- Run {run}/{RUNS} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X_full_binary,
            y_full_binary,
            test_size=0.3,
            random_state=42 + run,
            stratify=y_full_binary,
        )
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
        feature_file = os.path.join(DIRS["results_features"], f"selected_features_f{fsize}_run{run}.txt")
        with open(feature_file, "w") as f_out:
            f_out.write("\n".join(selected_features) + "\n")

        for model_name, clf_prototype in CLASSIFIERS.items():
            print(f"      Training {model_name}...")
            clf = clf_prototype
            try:
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)
                y_proba = clf.predict_proba(X_test_fs)[:, 1]
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba)
                fpr, tpr, _ = roc_curve(y_test, y_proba)

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
                roc_data_storage[(model_name, fsize)].append({"run": run, "fpr": fpr, "tpr": tpr, "auc": auc})

                model_filename = os.path.join(DIRS["models_classification"], f"{model_name}_f{fsize}_run{run}.pkl")
                with open(model_filename, "wb") as f_out:
                    pickle.dump({"model": clf, "selector": selector, "features": selected_features}, f_out)
            except Exception as e:
                print(f"      Error training/predicting {model_name} fsize={fsize}, run={run}: {e}")
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
                roc_data_storage[(model_name, fsize)].append({"run": run, "fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": np.nan})
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"Error training/predicting: model={model_name}, fsize={fsize}, run={run}, error={e}\n"
                    )

results_df = pd.DataFrame(classification_results_list)
results_csv = os.path.join(DIRS["results"], "classification_metrics_all_runs.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nSaved detailed classification metrics to: {results_csv}")

roc_data_file = os.path.join(DIRS["results"], "roc_data_all_runs.pkl")
with open(roc_data_file, "wb") as f:
    pickle.dump(roc_data_storage, f)
print(f"Saved ROC curve data to: {roc_data_file}")

with open(LOG_FILE, "a") as f:
    f.write("\nBinary Classification Loop Completed:\n")
    f.write(f"  Saved detailed metrics: {results_csv}\n  Saved ROC data: {roc_data_file}\n")
    f.write(f"  Saved models/selectors in: {DIRS['models_classification']}\n")
    f.write(f"  Saved selected features in: {DIRS['results_features']}\n")

print("\nBinary classification loop complete.")
