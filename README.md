# hnscc-analysis

This repository provides helper scripts for obtaining the HNSCC dataset used in
the ML Copilot Agent case study.

## Dataset

The processed HNSCC dataset files are included in this repository under the `data/` directory:

```
data/
├── cptac.survival.fcptac.csv
├── survival_cptac.csv
├── survival_tcga.csv
└── tcga.survival.fcptac.csv
```

The analysis scripts automatically locate these files, so no external download is required.

## Raw Predictions

After validation runs (Step 7), raw prediction CSVs for full TCGA and CPTAC are saved to both:
```
hnscc_robust_analysis_output/results/predictions/
```
and the user download folder (~Downloads/new_folder) for convenient access.

## Generating Additional Visualizations

After running the classification pipeline you can create several extra plots
(such as ROC grids, AUC boxplots and feature overlap visualisations) with:

```bash
python analysis_scripts/step11_extra_visualizations.py
```

The figures will be saved under `hnscc_robust_analysis_output/plots/classification_performance/`.
