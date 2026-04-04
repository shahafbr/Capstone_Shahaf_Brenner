import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# build_cv_dataframe - reads fold-level cross-validation results from all run folders and combines them into one long-format DataFrame.
def build_cv_dataframe(results_dir="results"):
    """
    Output columns:
        - model: model family name.
        - fold: fold number.
        - rmse: validation RMSE for that fold.
        - mae: validation MAE for that fold.
        - r2: validation R2 for that fold.
    """
    rows = []

    results_path = Path(results_dir)

    for family_dir in results_path.iterdir():
        if not family_dir.is_dir():
            continue

        for run_dir in family_dir.iterdir():
            if not run_dir.is_dir():
                continue

            cv_file = run_dir / "cv_results.csv"
            if not cv_file.exists():
                continue

            df = pd.read_csv(cv_file)

            # Extracting the model name from the run folder name (assuming format "ModelName_UUID"):
            model_name = run_dir.name.split("_")[0]

            # Getting the best row (based on RMSE ranking):
            if "rank_test_rmse" in df.columns:
                best_row = df.loc[df["rank_test_rmse"] == 1].iloc[0]
            else:
                # Fallback: use the first row if no ranking column exists.
                best_row = df.iloc[0]

            # Extracting fold-level test metrics:
            for i in range(5):  # assuming 5-fold CV
                rows.append({
                    "model": model_name,
                    "fold": i + 1,
                    # sklearn stores error scorers as negative values, so convert back to positive:
                    "rmse": -best_row[f"split{i}_test_rmse"],
                    "mae": -best_row[f"split{i}_test_mae"],

                    # R2 is already stored in its natural sign:
                    "r2": best_row[f"split{i}_test_r2"],
                })

    return pd.DataFrame(rows)

# plot_cv_boxplot - creates a boxplot of one CV metric across models.
# Useful for comparing median performance, spread, and outliers across folds.
def plot_cv_boxplot(df: pd.DataFrame, metric: str, save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    df.boxplot(column=metric, by="model")
    plt.title(f"Cross-Validation {metric.upper()} by Model")
    plt.suptitle("")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# plot_cv_line - creates a line plot of one CV metric across folds for each model.
# Useful for seeing how performance changes from fold to fold and whether models behave consistently.
def plot_cv_line(df: pd.DataFrame, metric: str, save_path: str) -> None:
    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        subset = df[df["model"] == model].sort_values("fold")
        plt.plot(subset["fold"], subset[metric], marker="o", label=model)

    plt.title(f"{metric.upper()} Across CV Folds")
    plt.xlabel("Fold")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# plot_cv_summary - creates a summary bar chart showing mean CV performance with standard deviation error bars for each model.
def plot_cv_summary(df: pd.DataFrame, metric: str, save_path: str) -> None:
    """
    Models are sorted so that:
        - lower mean is better for RMSE / MAE.
        - higher mean is better for R2.
    """
    summary = (
        df.groupby("model")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=(metric != "r2"))
    )

    plt.figure(figsize=(8, 5))
    plt.bar(summary["model"], summary["mean"], yerr=summary["std"], capsize=5)
    plt.title(f"Mean CV {metric.upper()} ± Std by Model")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()