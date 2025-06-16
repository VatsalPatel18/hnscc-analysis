# -*- coding: utf-8 -*-
"""Setup and configuration for HNSCC analysis pipeline."""
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

# Ensure lifelines is installed
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    print("ERROR: lifelines library is required. Install via 'pip install lifelines'.")
    sys.exit(1)
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning

# ===========================================================
# 0. Setup and Configuration
# ===========================================================
print("--- 0. Setting up Configuration ---")

# Ignore common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "data")

TCGA_EXPRESSION_FILE = os.path.join(DATA_DIR, "tcga.survival.fcptac.csv")
TCGA_SURVIVAL_FILE = os.path.join(DATA_DIR, "survival_tcga.csv")
CPTAC_EXPRESSION_FILE = os.path.join(DATA_DIR, "cptac.survival.fcptac.csv")
CPTAC_SURVIVAL_FILE = os.path.join(DATA_DIR, "survival_cptac.csv")

MAIN_OUTPUT_DIR = "hnscc_robust_analysis_output"
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

K_RANGE = range(2, 11)
CLUSTERING_METHODS = ["kmeans", "agglomerative", "gmm"]

FEATURE_SIZES = [2, 5, 8, 10, 12, 15]
RUNS = 100
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

P_VALUE_THRESHOLD_TCGA = 0.05
P_VALUE_THRESHOLD_CPTAC = 0.05
P_VALUE_THRESHOLD_CPTAC_RELAXED = 0.10

DIRS = {
    "data": os.path.join(MAIN_OUTPUT_DIR, "data"),
    "plots": os.path.join(MAIN_OUTPUT_DIR, "plots"),
    "models": os.path.join(MAIN_OUTPUT_DIR, "models"),
    "results": os.path.join(MAIN_OUTPUT_DIR, "results"),
    "plots_clustering": os.path.join(MAIN_OUTPUT_DIR, "plots", "clustering"),
    "plots_classification": os.path.join(MAIN_OUTPUT_DIR, "plots", "classification_performance"),
    "plots_validation": os.path.join(MAIN_OUTPUT_DIR, "plots", "survival_validation"),
    "models_clustering": os.path.join(MAIN_OUTPUT_DIR, "models", "clustering"),
    "models_classification": os.path.join(MAIN_OUTPUT_DIR, "models", "classification"),
    "results_features": os.path.join(MAIN_OUTPUT_DIR, "results", "feature_selection"),
    "results_predictions": os.path.join(MAIN_OUTPUT_DIR, "results", "predictions")
}
for path in DIRS.values():
    os.makedirs(path, exist_ok=True)

LOG_FILE = os.path.join(MAIN_OUTPUT_DIR, "analysis_log.txt")
with open(LOG_FILE, "w") as f:
    f.write(f"Analysis started at {pd.Timestamp.now()}\n")
    f.write(f"Main output directory: {MAIN_OUTPUT_DIR}\n")
    f.write(f"TCGA Expression: {TCGA_EXPRESSION_FILE}\n")
    f.write(f"TCGA Survival: {TCGA_SURVIVAL_FILE}\n")
    f.write(f"CPTAC Expression: {CPTAC_EXPRESSION_FILE}\n")
    f.write(f"CPTAC Survival: {CPTAC_SURVIVAL_FILE}\n")
    f.write(f"Clustering K range: {list(K_RANGE)}\n")
    f.write(f"Classification Feature Sizes: {FEATURE_SIZES}\n")
    f.write(f"Number of classification runs: {RUNS}\n")
    f.write(
        f"Significance Thresholds: TCGA P<{P_VALUE_THRESHOLD_TCGA}, CPTAC P<{P_VALUE_THRESHOLD_CPTAC} (Strict), CPTAC P<{P_VALUE_THRESHOLD_CPTAC_RELAXED} (Relaxed)\n"
    )

print(f"Setup complete. All outputs will be saved under: {MAIN_OUTPUT_DIR}")
