from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.synthetic_db.synthetic_config import (
    SyntheticScenarioConfig,
    INTRA_CATEGORY_RHO,
    NOISE_STD,
    OPEN_ENDED_RED_THRESHOLD,
)
from src.synthetic_db.synthetic_schema import (
    load_schema,
    compute_alpha,
    sample_and_normalize_weights,
    prepare_schema_for_scenario,
)
from src.synthetic_db.synthetic_sampling import (
    sample_behavioral_values,
    sample_structural_values,
    generate_uniform_copula,
)
from src.synthetic_db.synthetic_risk import compute_risk_intensity


# ====================================================================================
# SECTION 3.3.5 - Burnout Risk Construction:
#    - Equation (9): y = alpha * sum_core(w~_j z_j) + (1-alpha) * sum_sec(v~_k z_k).
#    - Equation (10): y* = y + epsilon.
#    - Equation (11): Risk Index = 100 * y*.
# ====================================================================================

def generate_dataset(
    schema: pd.DataFrame,
    n: int,
    data_seed: int,
    weight_seed: int | None = None,
    intra_category_rho: float = INTRA_CATEGORY_RHO,
    noise_std: float = NOISE_STD,
    alpha_override: float | None = None,
    open_ended_red_threshold: float = OPEN_ENDED_RED_THRESHOLD,
    include_latent_columns: bool = True,
) -> pd.DataFrame:
    rng_data = np.random.default_rng(data_seed)
    rng_weights = np.random.default_rng(weight_seed if weight_seed is not None else data_seed)

    # Eq. (7) - alpha computed from schema importance/overridden for sensitivity tests:
    alpha = compute_alpha(schema) if alpha_override is None else float(alpha_override)

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    # Eq. (8) + Eq. (5) - sampling raw weights then normalizing within groups to get w~_j and v~_k:
    weights = sample_and_normalize_weights(schema, rng_weights)
    schema_w = schema.merge(weights[["feature", "normalized_weight"]], on="feature", how="left")

    # Section 3.3.2 - Correlation Structure:
    U = generate_uniform_copula(schema_w, n=n, rho=intra_category_rho, rng=rng_data)

    # Section 3.3.1 + 3.3.4 - Feature Value Sampling and Risk Intensity Computation:
    X = pd.DataFrame(index=np.arange(n))
    Z = pd.DataFrame(index=np.arange(n))

    # Looping through features in schema order to maintain consistency:
    for _, row in schema_w.iterrows():
        feature = row["feature"]
        u = U[feature].values

        if row["feature_type"] == "behavioral":
            x = sample_behavioral_values(u, row, open_ended_red_threshold)
        elif row["feature_type"] == "structural":
            x = sample_structural_values(u, row, open_ended_red_threshold)
        else:
            raise ValueError(f"Unknown feature_type '{row['feature_type']}' for {feature}")

        z = compute_risk_intensity(x, row, open_ended_red_threshold)

        X[feature] = x
        Z[feature] = z

    # Equation (9) - additive burnout risk score before noise and scaling:
    core_feats = schema_w.loc[schema_w["group"] == "core", ["feature", "normalized_weight"]]
    sec_feats = schema_w.loc[schema_w["group"] == "secondary", ["feature", "normalized_weight"]]

    core_component = np.zeros(n)
    sec_component = np.zeros(n)

    for _, row in core_feats.iterrows():
        core_component += row["normalized_weight"] * Z[row["feature"]].values

    for _, row in sec_feats.iterrows():
        sec_component += row["normalized_weight"] * Z[row["feature"]].values

    y = alpha * core_component + (1.0 - alpha) * sec_component

    # Equation (10) - additive Gaussian noise:
    epsilon = rng_data.normal(loc=0.0, scale=noise_std, size=n)
    y_star = y + epsilon

    # Equation (11) - scale target to 0–100 and clip:
    risk_index = np.clip(100.0 * y_star, 0.0, 100.0)

    # Final dataset:
    out = X.copy()
    out["burnout_risk_index"] = risk_index

    if include_latent_columns:
        out["latent_y"] = y
        out["latent_y_star"] = y_star

    return out


# ============================================================ #
#           Scenario Execution and Scenario Saving:            #
# ============================================================ #

def get_scenario_output_dir(config: SyntheticScenarioConfig) -> Path:
    # Every scenario writes to its own directory to avoid overriding the current DB:
    return config.output_root / config.scenario_name


def save_scenario_metadata(
    config: SyntheticScenarioConfig,
    schema: pd.DataFrame,
    output_dir: Path,
) -> Path:
    # Every scenario saves its metadata to keep runs auditable:
    metadata = asdict(config)
    metadata["schema_path"] = str(config.schema_path)
    metadata["output_root"] = str(config.output_root)
    metadata["retained_features"] = schema["feature"].tolist()
    metadata["n_features"] = int(len(schema))
    metadata["computed_alpha_from_schema"] = float(compute_alpha(schema))

    meta_path = output_dir / config.metadata_filename
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def generate_scenario_datasets(config: SyntheticScenarioConfig) -> dict[str, pd.DataFrame]:
    # Returning both DBs in-memory so pipelines can "choose" to save, inspect, or pass paths onward:
    schema = load_schema(config.schema_path)
    schema = prepare_schema_for_scenario(schema, feature_drop=config.feature_drop)

    small_df = generate_dataset(
        schema=schema,
        n=config.small_n,
        data_seed=config.data_seed_small,
        weight_seed=config.weight_seed_small,
        intra_category_rho=config.intra_category_rho,
        noise_std=config.noise_std,
        alpha_override=config.alpha_override,
        open_ended_red_threshold=config.open_ended_red_threshold,
        include_latent_columns=config.include_latent_columns,
    )

    full_df = generate_dataset(
        schema=schema,
        n=config.full_n,
        data_seed=config.data_seed_full,
        weight_seed=config.weight_seed_full,
        intra_category_rho=config.intra_category_rho,
        noise_std=config.noise_std,
        alpha_override=config.alpha_override,
        open_ended_red_threshold=config.open_ended_red_threshold,
        include_latent_columns=config.include_latent_columns,
    )

    return {
        "schema": schema,
        "small": small_df,
        "full": full_df,
    }


def generate_and_save_scenario(config: SyntheticScenarioConfig) -> dict[str, Path]:
    # The main high-level entry point for future pipeline files:
    output_dir = get_scenario_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_scenario_datasets(config)
    schema = generated["schema"]
    small_df = generated["small"]
    full_df = generated["full"]

    small_path = output_dir / f"synthetic_small_{config.small_n}.csv"
    full_path = output_dir / f"synthetic_full_{config.full_n}.csv"

    small_df.to_csv(small_path, index=False)
    full_df.to_csv(full_path, index=False)

    meta_path = save_scenario_metadata(config, schema, output_dir)

    return {
        "scenario_dir": output_dir,
        "small_path": small_path,
        "full_path": full_path,
        "metadata_path": meta_path,
    }


# ============================================================ #
#         Main Execution (Saving the 2 datasets):              #
# ============================================================ #

def main():
    # The main high-level entry point for future pipeline files:
    config = SyntheticScenarioConfig()

    saved = generate_and_save_scenario(config)

    small_df = pd.read_csv(saved["small_path"])
    full_df = pd.read_csv(saved["full_path"])

    print(f"Saved small dataset to: {saved['small_path']}")
    print(f"Saved full dataset to: {saved['full_path']}")
    print(f"Saved scenario metadata to: {saved['metadata_path']}")

    print("\nQuick summary:")
    print(f"Small shape: {small_df.shape}")
    print(f"Full shape:  {full_df.shape}")
    print(f"Risk range (small): {small_df['burnout_risk_index'].min():.2f} - {small_df['burnout_risk_index'].max():.2f}")
    print(f"Risk range (full):  {full_df['burnout_risk_index'].min():.2f} - {full_df['burnout_risk_index'].max():.2f}")


if __name__ == "__main__":
    main()