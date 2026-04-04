from pathlib import Path
import json
import ast

import pandas as pd
import matplotlib.pyplot as plt

# _safe_read_json - loads a JSON file from disk using UTF-8 encoding:
def _safe_read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# # _safe_parse_params - attempts to safely parse parameter strings into Python objects:
def _safe_parse_params(x):
    # If the input is NaN, return an empty dictionary:
    if pd.isna(x):
        return {}
    # If the input is already a dictionary, return it as-is:
    if isinstance(x, dict):
        return x
    # If the input is a string, attempt to parse it as a Python literal (e.g., dict, list):
    try:
        return ast.literal_eval(x)
    # If parsing fails, return the raw string in a dictionary for reference:
    except Exception:
        return {"raw": str(x)}

# _find_run_dirs - scans the results root and returns all run directories.
# Expected structure:
"""
     results/
        linear/
            OLS_xxxxxxxx/
        regularized/
            Lasso_xxxxxxxx/
            Ridge_xxxxxxxx/
        tree/
            GradientBoosting_xxxxxxxx/
            RandomForest_xxxxxxxx/
"""
def _find_run_dirs(results_root="results"):
    results_root = Path(results_root)
    run_dirs = []

    # Iterate through the results directory, looking for subdirectories that represent individual runs:
    for family_dir in results_root.iterdir():
        if not family_dir.is_dir():
            continue
        # Within each model family directory, look for run directories (which should also be directories):
        for run_dir in family_dir.iterdir():
            if run_dir.is_dir():
                run_dirs.append(run_dir)

    return run_dirs

# _extract_model_name - extracts the model family name from the run directory name.
def _extract_model_name(run_dir: Path):
    """
    Example: GradientBoosting_35fbf139 -> GradientBoosting  
    """
    return run_dir.name.split("_")[0]


def load_test_metrics(results_root="results") -> pd.DataFrame:
    """
    Build a dataframe from each run's test_metrics.json and run_config.json.
    """
    rows = []

    for run_dir in _find_run_dirs(results_root):
        metrics_file = run_dir / "test_metrics.json"
        config_file = run_dir / "run_config.json"

        # If the test_metrics.json file doesn't exist, skip this run:
        if not metrics_file.exists():
            continue

        metrics = _safe_read_json(metrics_file)

        # Attempt to read best_params from the config file, but if it doesn't exist or fails to parse, just use an empty dictionary:
        params = {}
        if config_file.exists():
            config = _safe_read_json(config_file)
            params = config.get("best_params", {}) or {}

        # Append a row to the results dataframe with the model name, run ID, test metrics, best parameters, and run path for reference:
        rows.append(
            {
                "model": _extract_model_name(run_dir),
                "run_id": run_dir.name,
                "rmse_test": metrics.get("rmse"),
                "mae_test": metrics.get("mae"),
                "r2_test": metrics.get("r2"),
                "best_params": params,
                "run_path": str(run_dir),
            }
        )

    # Convert the list of rows into a DataFrame, and sort by R2 test score in descending order for easier analysis:
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No test_metrics.json files were found.")

    return df.sort_values("r2_test", ascending=False).reset_index(drop=True)


def load_cv_summary(results_root="results") -> pd.DataFrame:
    """
    Extract fold-level CV metrics for the BEST hyperparameter row from cv_results.csv,
    then summarize mean/std per model.
    """
    rows = []

    # Iterate through each run directory, looking for cv_results.csv files:
    for run_dir in _find_run_dirs(results_root):
        cv_file = run_dir / "cv_results.csv"
        if not cv_file.exists():
            continue

        df = pd.read_csv(cv_file)
        model_name = _extract_model_name(run_dir)

        # Get the best row based on RMSE ranking (if available), otherwise just take the first row as a fallback:
        if "rank_test_rmse" in df.columns:
            best_row = df.loc[df["rank_test_rmse"] == 1].iloc[0]
        else:
            best_row = df.iloc[0]

        # Dynamically detect fold indices based on column names, allowing for any number of folds as long as they follow the naming pattern:
        fold_indices = sorted(
            {
                int(col.replace("split", "").replace("_test_rmse", ""))
                for col in df.columns
                if col.startswith("split") and col.endswith("_test_rmse")
            }
        )

        # Extract the CV metrics for each fold and append them to the list of rows, converting negative RMSE/MAE back to positive values:
        for i in fold_indices:
            rows.append(
                {
                    "model": model_name,
                    "fold": i + 1,
                    "rmse_cv": -best_row[f"split{i}_test_rmse"],
                    "mae_cv": -best_row[f"split{i}_test_mae"],
                    "r2_cv": best_row[f"split{i}_test_r2"],
                }
            )

    cv_df = pd.DataFrame(rows)

    if cv_df.empty:
        raise ValueError("No cv_results.csv files were found.")

    summary = (
        cv_df.groupby("model")
        .agg(
            rmse_cv_mean=("rmse_cv", "mean"),
            rmse_cv_std=("rmse_cv", "std"),
            mae_cv_mean=("mae_cv", "mean"),
            mae_cv_std=("mae_cv", "std"),
            r2_cv_mean=("r2_cv", "mean"),
            r2_cv_std=("r2_cv", "std"),
        )
        .reset_index()
    )

    return summary, cv_df


def build_final_comparison_table(results_root="results") -> pd.DataFrame:
    """
    Merge test metrics + CV summary into one thesis-ready table.
    """
    test_df = load_test_metrics(results_root)
    cv_summary_df, _ = load_cv_summary(results_root)

    # Merge the test metrics dataframe with the CV summary dataframe on the model name, using a left join to keep all test results even if CV summary is missing for some models:
    final_df = test_df.merge(cv_summary_df, on="model", how="left")

    # Convert the best_params column from stringified dictionaries to actual dictionaries for better readability and potential further analysis:
    final_df["best_params_str"] = final_df["best_params"].apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )

    # Define the desired column order for the final comparison table, ensuring that key metrics and information are presented in a logical and consistent manner:
    ordered_cols = [
        "model",
        "run_id",
        "best_params_str",
        "rmse_test",
        "mae_test",
        "r2_test",
        "rmse_cv_mean",
        "rmse_cv_std",
        "mae_cv_mean",
        "mae_cv_std",
        "r2_cv_mean",
        "r2_cv_std",
        "run_path",
    ]

    # Return the final comparison table with the specified column order, sorted by R2 test score in descending order for easier analysis and presentation:
    return final_df[ordered_cols].sort_values("r2_test", ascending=False).reset_index(drop=True)


def save_final_comparison_table(results_root="results", output_dir="results/final_analysis"):
    """
    - Function: Save final comparison table to disk as a CSV file for use in the thesis. 

    Inputs:
        results_root (str, optional): _description_. Defaults to "results".
        output_dir (str, optional): _description_. Defaults to "results/final_analysis".

    Outputs:
        _type_: _description_
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = build_final_comparison_table(results_root)

    csv_path = output_dir / "final_model_comparison.csv"
    final_df.to_csv(csv_path, index=False)

    return final_df, csv_path


def load_feature_importance(results_root="results") -> pd.DataFrame:
    """
    Load and combine feature_importance.csv across models.

    Expected columns may vary, so this function tries to normalize:
    - feature / Feature / variable
    - importance / coefficient / abs_coefficient / score
    """
    rows = []

    # Iterate through each run directory, looking for feature_importance.csv files:
    for run_dir in _find_run_dirs(results_root):
        fi_file = run_dir / "feature_importance.csv"
        if not fi_file.exists():
            continue

        df = pd.read_csv(fi_file)
        model_name = _extract_model_name(run_dir)

        # Create a lowercase mapping of column names to handle variations in naming conventions across different feature importance outputs:
        lower_map = {c.lower(): c for c in df.columns}

        feature_col = None
        for candidate in ["feature", "features", "variable", "predictor", "name"]:
            if candidate in lower_map:
                feature_col = lower_map[candidate]
                break

        value_col = None
        for candidate in [
            "importance",
            "coefficient",
            "abs_coefficient",
            "score",
            "weight",
            "gain",
        ]:
            if candidate in lower_map:
                value_col = lower_map[candidate]
                break

        if feature_col is None:
            raise ValueError(
                f"Could not detect feature column in {fi_file}. Columns: {list(df.columns)}"
            )

        if value_col is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError(
                    f"Could not detect numeric importance column in {fi_file}. Columns: {list(df.columns)}"
                )
            value_col = numeric_cols[0]

        # Extract the relevant columns, normalize the importance values to be comparable across models, and append them to the list of rows:
        tmp = df[[feature_col, value_col]].copy()
        tmp.columns = ["feature", "importance_raw"]
        tmp["model"] = model_name

        # Use absolute magnitude so linear coefficients can be compared with tree-based importance scores on a common non-signed scale:
        tmp["importance"] = tmp["importance_raw"].abs()

        rows.append(tmp[["model", "feature", "importance"]])

    if not rows:
        raise ValueError("No feature_importance.csv files were found.")

    # Combine all the individual feature importance dataframes into one comprehensive dataframe for analysis and visualization:
    combined = pd.concat(rows, ignore_index=True)

    return combined


def save_feature_importance_outputs(
    results_root="results",
    output_dir="results/final_analysis",
    top_n=10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fi_df = load_feature_importance(results_root)

    # normalize per model to 0..1 for easier comparison across model types:
    fi_df["importance_norm"] = fi_df.groupby("model")["importance"].transform(
        lambda s: s / s.max() if s.max() != 0 else s
    )

    fi_df.to_csv(output_dir / "all_feature_importance.csv", index=False)

    # Averaging the normalized importance across models:
    avg_fi = (
        fi_df.groupby("feature")
        .agg(mean_importance_norm=("importance_norm", "mean"))
        .reset_index()
        .sort_values("mean_importance_norm", ascending=False)
    )

    # Select the top N features based on average normalized importance across all models:
    top_features = avg_fi.head(top_n)["feature"].tolist()

    # Create a pivot table for the top features, showing their normalized importance across different models, and save it to disk for reference:
    top_df = fi_df[fi_df["feature"].isin(top_features)].copy()

    pivot_df = (
        top_df.pivot_table(
            index="feature",
            columns="model",
            values="importance_norm",
            aggfunc="mean",
        )
        .fillna(0)
    )

    # Ordering rows by average importance:
    pivot_df["avg"] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values("avg", ascending=True)
    pivot_df = pivot_df.drop(columns=["avg"])

    pivot_path = output_dir / "feature_importance_top_features.csv"
    pivot_df.to_csv(pivot_path)

    # Plotting the top features across models:
    plt.figure(figsize=(10, 7))
    pivot_df.plot(kind="barh", figsize=(10, 7))
    plt.title(f"Top {top_n} Features by Normalized Importance Across Models")
    plt.xlabel("Normalized Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plot_path = output_dir / "feature_importance_top_features.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return fi_df, pivot_df, plot_path


def find_best_model_run(results_root="results", metric="r2_test"):
    """
    Return the best run path based on final comparison table. By default, the best model is selected using held-out test R2.
    """
    final_df = build_final_comparison_table(results_root)

    if metric not in final_df.columns:
        raise ValueError(f"Metric {metric} not found in final comparison table.")

    # For error metrics, lower is better.
    # For R2, higher is better.
    ascending = metric in ["rmse_test", "mae_test"]
    best_row = final_df.sort_values(metric, ascending=ascending).iloc[0]

    # Return the path to the best model's run directory and the model name for reference:
    return Path(best_row["run_path"]), best_row["model"]


def load_predictions_for_best_model(results_root="results", metric="r2_test") -> pd.DataFrame:
     # Locate the best model run based on the requested metric:
    best_run_dir, model_name = find_best_model_run(results_root, metric=metric)

    pred_file = best_run_dir / "test_predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_file}")

    df = pd.read_csv(pred_file)

    lower_map = {c.lower(): c for c in df.columns}

    actual_col = None
    pred_col = None

    # Detecting the column containing the true target values:
    for candidate in ["y_true", "actual", "target", "true", "observed"]:
        if candidate in lower_map:
            actual_col = lower_map[candidate]
            break

    # Detecting the column containing the predicted values:
    for candidate in ["y_pred", "predicted", "prediction", "pred", "estimate"]:
        if candidate in lower_map:
            pred_col = lower_map[candidate]
            break

    if actual_col is None or pred_col is None:
        raise ValueError(
            f"Could not detect actual/predicted columns in {pred_file}. "
            f"Columns: {list(df.columns)}"
        )

    # Standardizing the column names and compute diagnostic error fields such as residuals and absolute errors for further analysis and visualization:
    out = df[[actual_col, pred_col]].copy()
    out.columns = ["actual", "predicted"]
    out["residual"] = out["actual"] - out["predicted"]
    out["abs_error"] = out["residual"].abs()
    out["model"] = model_name

    # Return the predictions dataframe along with the model name for reference in diagnostics and plotting:
    return out, model_name


def save_prediction_diagnostics(
    results_root="results",
    output_dir="results/final_analysis",
    metric="r2_test",
):
    
    # Creating a output folder for prediction-diagnostic artifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df, model_name = load_predictions_for_best_model(results_root, metric=metric)

    # Saving the row-level diagnostic values to a CSV file for potential inclusion in the thesis or further analysis:
    pred_df.to_csv(output_dir / f"{model_name}_prediction_diagnostics.csv", index=False)

    # 1. Actual vs Predicted plot:
    # Used to assess how closely predictions follow the true target values. A perfect model would have all points along the diagonal line.
    plt.figure(figsize=(7, 7))
    plt.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.7)
    min_val = min(pred_df["actual"].min(), pred_df["predicted"].min())
    max_val = max(pred_df["actual"].max(), pred_df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Residual histogram plot:
    # Used to assess the distribution of prediction errors. Ideally, residuals should be symmetrically distributed around zero.
    plt.hist(pred_df["residual"], bins=20)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution ({model_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_residual_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Residuals vs Predicted plot:
    # Used to assess the relationship between predicted values and prediction errors.
    plt.figure(figsize=(8, 5))
    plt.scatter(pred_df["predicted"], pred_df["residual"], alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs Predicted ({model_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_residuals_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Return the predictions dataframe and model name for potential further use in the thesis or additional analyses:
    return pred_df, model_name


def run_all_final_analysis(results_root="results", output_dir="results/final_analysis", top_n=10):
    # Running the full final-analysis pipeline:
    #   1. model comparison table
    #   2. feature importance outputs
    #   3. best-model prediction diagnostics
    final_df, final_csv = save_final_comparison_table(results_root, output_dir)
    fi_df, fi_pivot_df, fi_plot = save_feature_importance_outputs(results_root, output_dir, top_n=top_n)
    pred_df, best_model_name = save_prediction_diagnostics(results_root, output_dir, metric="r2_test")

    return {
        "final_comparison_table": str(final_csv),
        "feature_importance_plot": str(fi_plot),
        "best_model_for_diagnostics": best_model_name,
        "n_models": final_df["model"].nunique(),
        "n_prediction_rows": len(pred_df),
    }