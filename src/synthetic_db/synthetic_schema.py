from __future__ import annotations

import pandas as pd
import numpy as np

from src.synthetic_db.synthetic_config import FEATURE_TYPE, DIRECTION, CATEGORY


# =========================================================== #
#             Loading and Validating the schema:              #
# =========================================================== #

def load_schema(schema_path) -> pd.DataFrame:
    df = pd.read_csv(schema_path)

    required_cols = [
        "feature",
        "importance_mean",
        "importance_std",
        "green_low",
        "green_high",
        "red_low",
        "red_high",
        "group",
    ]
    missing = [c for c in required_cols if c not in df.columns]

    # basic validation of schema values:
    if missing:
        raise ValueError(f"Schema is missing required columns: {missing}")

    if not (df["green_low"] < df["green_high"]).all():
        raise ValueError("Invalid green zone found in schema.")
    if not (df["red_low"] < df["red_high"]).all():
        raise ValueError("Invalid red zone found in schema.")

    # Attaching manual survey metadata:
    df["feature_type"] = df["feature"].map(FEATURE_TYPE)
    df["direction"] = df["feature"].map(DIRECTION)
    df["category"] = df["feature"].map(CATEGORY)

    for col in ["feature_type", "direction", "category"]:
        missing_meta = df[df[col].isna()]["feature"].tolist()
        if missing_meta:
            raise ValueError(f"Missing {col} mapping for features: {missing_meta}")

    return df.copy()


# ============================================================ #
#      Computing ALPHA - Section 3.3.5 / Equation (7):         #
# ============================================================ #

def compute_alpha(schema: pd.DataFrame) -> float:
    core_sum = schema.loc[schema["group"] == "core", "importance_mean"].sum()
    total_sum = schema["importance_mean"].sum()
    if total_sum <= 0:
        raise ValueError("Total importance sum must be positive.")
    return float(core_sum / total_sum)


# ============================================================================ #
# Equations (6 & 8) - sample weights then normalize within groups (Equation 5) #
# ============================================================================ #

def sample_and_normalize_weights(schema: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sampled = schema["importance_mean"].values + rng.normal(
        loc=0.0,
        scale=schema["importance_std"].values,
        size=len(schema)
    )

    # Ensures that the weights are positive:
    sampled = np.clip(sampled, 1e-6, None)

    out = schema[["feature", "group"]].copy()
    out["sampled_weight"] = sampled

    # Equation (5) - normalize weights within group:
    out["normalized_weight"] = 0.0
    for grp in ["core", "secondary"]:
        mask = out["group"] == grp
        if mask.any():
            denom = out.loc[mask, "sampled_weight"].sum()
            out.loc[mask, "normalized_weight"] = out.loc[mask, "sampled_weight"] / denom

    return out


# =========================================================== #
#             Scenario-Level Schema Preparation:              #
# =========================================================== #

def prepare_schema_for_scenario(
    schema: pd.DataFrame,
    feature_drop: list[str] | None = None,
) -> pd.DataFrame:
    # Keeping the feature-removal tests consistent by removing dropped features before both x and y are built:
    feature_drop = feature_drop or []
    if not feature_drop:
        return schema.copy()

    out = schema.loc[~schema["feature"].isin(feature_drop)].copy()

    if out.empty:
        raise ValueError("All features were dropped. Scenario must retain at least one feature.")

    remaining_groups = set(out["group"].unique())
    if "core" not in remaining_groups or "secondary" not in remaining_groups:
        raise ValueError(
            "Feature drop removed an entire group. Circularity tests must retain both core and secondary features."
        )

    return out.reset_index(drop=True)