import os
from pathlib import Path
import pandas as pd


# ====================================================================== #
#                       Loading Expert Survey Data                       #
# ====================================================================== #

# Input files - expert ratings and zone definitions:
IMPORTANCE_PATH = Path("data\\split_by_stable_keys\\Importance.xlsx")
FREQUENCY_PATH = Path("data\\split_by_stable_keys\\Frequency.xlsx")
GREEN_ZONE_PATH = Path("data\\split_by_stable_keys\\Green_Zone.xlsx")
RED_ZONE_PATH = Path("data\\split_by_stable_keys\\Red_Zone.xlsx")

# Output directory:
OUTPUT_DIR = Path("data\\Expert_Aggregated_DB\\")


# ====================================================================== #
#                           Helper Functions                             #
# ====================================================================== #

def validate_input_file(path: Path) -> None:
    """
    - Function: Raise an error if the input file does not exist (Debugging).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    - Function: Load an Excel file and drop the expert_id column.
    - Assumptions: first column is expert_id, and remaining columns are feature names.
    """
    df = pd.read_excel(path)

    # Validation - Check for 'expert_id' column (Debugging):
    if "expert_id" not in df.columns:
        raise ValueError(f"'expert_id' column not found in file: {path}")

    # Drop 'expert_id' and return only feature columns:
    df_features = df.drop(columns=["expert_id"]).copy()
    return df_features


def validate_matching_columns(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> None:
    """
    - Function: Ensure two dataframes have the exact same feature columns in the same order (validation).
    """
    if list(df1.columns) != list(df2.columns):
        raise ValueError(
            f"Feature columns do not match between {name1} and {name2}.\n"
            f"{name1}: {list(df1.columns)}\n"
            f"{name2}: {list(df2.columns)}"
        )


def split_range_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Function: Split range strings of the form 'x,y' into numeric low/high columns.

        Input:                  Output:
            feature_a           feature_a_low  feature_a_high
            1,3                 1              3
            2,4                 2              4
    """
    # Initialize the result DataFrame:
    result = pd.DataFrame(index=df.index)

    # For each column, split the 'x,y' string into two numeric columns:
    for col in df.columns:
        split_vals = df[col].astype(str).str.split("|", expand=True)

        # Validation - Check that we have exactly two parts after splitting (Debugging):
        if split_vals.shape[1] != 2:
            raise ValueError(
                f"Column '{col}' does not contain valid 'x,y' formatted values."
            )

        # Convert the split values to numeric, coercing errors to NaN:
        result[f"{col}_low"] = pd.to_numeric(split_vals[0].str.strip(), errors="coerce")
        result[f"{col}_high"] = pd.to_numeric(split_vals[1].str.strip(), errors="coerce")

    return result


def build_zone_summary(zone_split_df: pd.DataFrame, zone_prefix: str) -> pd.DataFrame:
    """
    - Function: Convert split low/high columns into a tidy one-row-per-feature summary.

    Example output:
        feature     green_low   green_high
        feature_a   1.25        3.50
        feature_b   2.10        4.90
    """
    # Compute mean and std for each low/high column:
    zone_means = zone_split_df.mean()
    zone_stds = zone_split_df.std()

    # Extract unique feature names by removing the '_low'/'_high' suffix:
    features = sorted(set(col.rsplit("_", 1)[0] for col in zone_means.index))
    rows = []

    # Build a summary row for each feature:
    for feature in features:
        rows.append({
            "feature": feature,
            f"{zone_prefix}_low": zone_means[f"{feature}_low"],
            f"{zone_prefix}_high": zone_means[f"{feature}_high"],
            f"{zone_prefix}_low_std": zone_stds[f"{feature}_low"],
            f"{zone_prefix}_high_std": zone_stds[f"{feature}_high"],
        })

    return pd.DataFrame(rows)


def build_feature_schema(
    importance_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    green_df: pd.DataFrame,
    red_df: pd.DataFrame,
    core_fraction: float = 0.25
) -> tuple[pd.DataFrame, float]:
    """
    - Function: Build the aggregated feature-level schema for synthetic data generation.

    - Returns:
        1.feature_schema: DataFrame
        2.alpha: float
    """

    # Importance statistics:
    importance_mean = importance_df.mean()
    importance_std = importance_df.std()

    importance_stats = pd.DataFrame({
        "feature": importance_mean.index,
        "importance_mean": importance_mean.values,
        "importance_std": importance_std.values
    })

    # Frequency statistics:
    frequency_mean = frequency_df.mean()
    frequency_std = frequency_df.std()

    frequency_stats = pd.DataFrame({
        "feature": frequency_mean.index,
        "frequency_mean": frequency_mean.values,
        "frequency_std": frequency_std.values
    })

    # Green / Red zone statistics:
    green_split = split_range_columns(green_df)
    red_split = split_range_columns(red_df)

    green_stats = build_zone_summary(green_split, "green")
    red_stats = build_zone_summary(red_split, "red")

    # Merging all summaries:
    feature_schema = importance_stats.merge(frequency_stats, on="feature")
    feature_schema = feature_schema.merge(green_stats, on="feature")
    feature_schema = feature_schema.merge(red_stats, on="feature")

    # Remove low-importance features:
    importance_threshold = 3.0

    feature_schema = feature_schema[
        feature_schema["importance_mean"] > importance_threshold
    ].copy()

    # Sort by importance:
    feature_schema = feature_schema.sort_values(
        by="importance_mean",
        ascending=False
    ).reset_index(drop=True)

    # Define Core / Secondary:
    core_cutoff = int(len(feature_schema) * core_fraction)

    feature_schema["group"] = "secondary"
    feature_schema.loc[:core_cutoff - 1, "group"] = "core"

    # Compute alpha:
    core_sum = feature_schema.loc[feature_schema["group"] == "core", "importance_mean"].sum()
    total_sum = feature_schema["importance_mean"].sum()

    alpha = core_sum / total_sum

    # Validation flags:
    feature_schema["green_valid"] = feature_schema["green_low"] < feature_schema["green_high"]
    feature_schema["red_valid"] = feature_schema["red_low"] < feature_schema["red_high"]

    # Overlap check:
    # True -> green upper bound is above red lower bound. This is not always "wrong", but useful for inspection.
    feature_schema["green_red_overlap"] = feature_schema["green_high"] > feature_schema["red_low"]

    return feature_schema, alpha


def save_outputs(feature_schema: pd.DataFrame, alpha: float, output_dir: Path) -> None:
    """
    - Function: Save aggregated schema and alpha to disk.
    """
    # Ensure output directory exists:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature schema and alpha value to files:
    schema_csv_path = output_dir / "aggregated_feature_schema.csv"
    alpha_txt_path = output_dir / "alpha_value.txt"

    feature_schema.to_csv(schema_csv_path, index=False)

    with open(alpha_txt_path, "w", encoding="utf-8") as f:
        f.write(f"alpha={alpha:.6f}\n")

    print(f"Saved aggregated feature schema to: {schema_csv_path}")
    print(f"Saved alpha value to: {alpha_txt_path}")


# ====================================================================== #
#                                MAIN                                    #
# ====================================================================== #

def main() -> None:
    # Validating files:
    validate_input_file(IMPORTANCE_PATH)
    validate_input_file(FREQUENCY_PATH)
    validate_input_file(GREEN_ZONE_PATH)
    validate_input_file(RED_ZONE_PATH)

    # Loading the data:
    importance_df = load_feature_matrix(IMPORTANCE_PATH)
    frequency_df = load_feature_matrix(FREQUENCY_PATH)
    green_df = load_feature_matrix(GREEN_ZONE_PATH)
    red_df = load_feature_matrix(RED_ZONE_PATH)

    # Validating matching feature columns:
    validate_matching_columns(importance_df, frequency_df, "Importance", "Frequency")
    validate_matching_columns(importance_df, green_df, "Importance", "Green Zone")
    validate_matching_columns(importance_df, red_df, "Importance", "Red Zone")

    # Building aggregated feature schema:
    feature_schema, alpha = build_feature_schema(
        importance_df=importance_df,
        frequency_df=frequency_df,
        green_df=green_df,
        red_df=red_df,
        core_fraction=0.25
    )

    # Printing quick summary:
    print("\nAggregated Feature Schema:")
    print(feature_schema)

    print("\nShape:")
    print(feature_schema.shape)

    print("\nCore Features:")
    print(feature_schema.loc[feature_schema["group"] == "core", "feature"].tolist())

    print(f"\nAlpha: {alpha:.6f}")

    # Saving outputs:
    save_outputs(feature_schema, alpha, OUTPUT_DIR)


if __name__ == "__main__":
    main()