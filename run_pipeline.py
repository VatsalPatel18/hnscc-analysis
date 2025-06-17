import subprocess
import sys


def run_script(path):
    """Run a single analysis step."""
    print(f"\n==> Running {path}...")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {path}")


def main():
    steps = [
        "analysis_scripts/step0_setup.py",
        "analysis_scripts/step1_load_tcga.py",
        "analysis_scripts/step2_helper_functions.py",
        "analysis_scripts/step3_initial_clustering.py",
        "analysis_scripts/step4_prepare_binary_classification.py",
        "analysis_scripts/step5_classification_loop.py",
        "analysis_scripts/classification3_5_metrics_plots.py",
        "analysis_scripts/step6_aggregate_visualize.py",
        "analysis_scripts/step7_validation_summary.py",
        "analysis_scripts/step8_select_best_model.py",
        "analysis_scripts/step9_final_km_plots.py",
        "analysis_scripts/step10_reclustering.py",
        "analysis_scripts/step11_extra_visualizations.py",
    ]

    for step in steps:
        run_script(step)

    print("\nAll analysis steps completed.")


if __name__ == "__main__":
    main()
