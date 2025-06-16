# hnscc-analysis

This repository provides helper scripts for obtaining the HNSCC dataset used in
the ML Copilot Agent case study.

## Downloading the dataset

Run `scripts/download_dataset.sh` to clone the dataset from Hugging Face:

```bash
./scripts/download_dataset.sh
```

The paths to the processed data files are defined in `scripts/data_paths.py`.

## Generating Additional Visualizations

After running the classification pipeline you can create several extra plots
(such as ROC grids, AUC boxplots and feature overlap visualisations) with:

```bash
python analysis_scripts/step11_extra_visualizations.py
```

The figures will be saved under `hnscc_robust_analysis_output/plots/classification_performance/`.
