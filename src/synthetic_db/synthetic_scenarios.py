from __future__ import annotations

from src.synthetic_db.synthetic_config import (
    SyntheticScenarioConfig,
    RANDOM_SEED,
    INTRA_CATEGORY_RHO,
    NOISE_STD,
)
from src.synthetic_db.synthetic_schema import load_schema, compute_alpha


# baseline scenario - for the actual DB generation used by the main pipelines:
def build_baseline_scenario() -> SyntheticScenarioConfig:
    return SyntheticScenarioConfig(
        scenario_name="baseline",
        data_seed_small=RANDOM_SEED,
        data_seed_full=RANDOM_SEED + 1,
        weight_seed_small=RANDOM_SEED,
        weight_seed_full=RANDOM_SEED + 1,
    )


# no-correlation scenario - Correlation Structure Sensitivity (3.6.6):
def build_no_correlation_scenario() -> SyntheticScenarioConfig:
    return SyntheticScenarioConfig(
        scenario_name="no_correlation",
        intra_category_rho=0.0,
        data_seed_small=RANDOM_SEED,
        data_seed_full=RANDOM_SEED + 1,
        weight_seed_small=RANDOM_SEED,
        weight_seed_full=RANDOM_SEED + 1,
    )


# low-noise sensitivity scenario - Noise Sensitivity Analysis (3.6.4):
def build_low_noise_scenario() -> SyntheticScenarioConfig:
    return SyntheticScenarioConfig(
        scenario_name="low_noise",
        noise_std=0.025,
        data_seed_small=RANDOM_SEED,
        data_seed_full=RANDOM_SEED + 1,
        weight_seed_small=RANDOM_SEED,
        weight_seed_full=RANDOM_SEED + 1,
    )


# high-noise sensitivity scenario - Noise Sensitivity Analysis (3.6.4):
def build_high_noise_scenario() -> SyntheticScenarioConfig:
    return SyntheticScenarioConfig(
        scenario_name="high_noise",
        noise_std=0.10,
        data_seed_small=RANDOM_SEED,
        data_seed_full=RANDOM_SEED + 1,
        weight_seed_small=RANDOM_SEED,
        weight_seed_full=RANDOM_SEED + 1,
    )


# alpha-plus scenario (alpha + 10%) - Alpha Parameter Sensitivity (3.6.3):
def build_alpha_plus_10_scenario() -> SyntheticScenarioConfig:
    cfg = SyntheticScenarioConfig(scenario_name="alpha_plus_10")
    schema = load_schema(cfg.schema_path)
    base_alpha = compute_alpha(schema)
    cfg.alpha_override = min(1.0, base_alpha * 1.10)
    return cfg


# alpha-minus scenario (alpha - 10%) - Alpha Parameter Sensitivity (3.6.3):
def build_alpha_minus_10_scenario() -> SyntheticScenarioConfig:
    cfg = SyntheticScenarioConfig(scenario_name="alpha_minus_10")
    schema = load_schema(cfg.schema_path)
    base_alpha = compute_alpha(schema)
    cfg.alpha_override = max(0.0, base_alpha * 0.90)
    return cfg


# feature-drop scenario - Feature Contribution Stability (3.6.5):
def build_feature_drop_scenario(feature_name: str) -> SyntheticScenarioConfig:
    return SyntheticScenarioConfig(
        scenario_name=f"drop_{feature_name}",
        feature_drop=[feature_name],
        data_seed_small=RANDOM_SEED,
        data_seed_full=RANDOM_SEED + 1,
        weight_seed_small=RANDOM_SEED,
        weight_seed_full=RANDOM_SEED + 1,
    )


# weight uncertainty scenario - Weight Uncertainty Sensitivity (3.6.2):
def build_weight_resample_scenario(run_idx: int) -> SyntheticScenarioConfig:
    return SyntheticScenarioConfig(
        scenario_name=f"weight_resample_{run_idx:02d}",
        intra_category_rho=INTRA_CATEGORY_RHO,
        noise_std=NOISE_STD,
        data_seed_small=RANDOM_SEED,
        data_seed_full=RANDOM_SEED + 1,
        weight_seed_small=RANDOM_SEED + 100 + run_idx,
        weight_seed_full=RANDOM_SEED + 200 + run_idx,
    )