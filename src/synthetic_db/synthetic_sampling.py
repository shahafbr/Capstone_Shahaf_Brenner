from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm, uniform

from src.synthetic_db.synthetic_config import NON_NEGATIVE_FEATURES


def apply_feature_bounds(feature: str, lower: float, upper: float) -> tuple[float, float]:
    """
    Function - Apply feature-specific bounds to the sampling support.
    """
    if NON_NEGATIVE_FEATURES.get(feature, False):
        lower = max(0.0, lower)

    if upper <= lower:
        upper = lower + 1e-6

    return lower, upper


# ============================================================ #
#              Open-Ended Red Zone Cap Handling:               #
# ============================================================ #

def effective_red_high(row: pd.Series, open_ended_red_threshold: float) -> float:
    red_high = row["red_high"]
    red_low = row["red_low"]

    green_width = max(row["green_high"] - row["green_low"], 1e-6)

    if red_high >= open_ended_red_threshold:
        # treating red_low as threshold and creating a finite modeling cap:
        return red_low + 3.0 * green_width
    return red_high


# ============================================================
# SECTION 3.3.1 - Feature Value Sampling:
#    - Behavioral -> truncated normal centered in green zone,
#    - Structural -> uniform over plausible range.
# ============================================================

def sample_behavioral_values(
    u: np.ndarray,
    row: pd.Series,
    open_ended_red_threshold: float,
) -> np.ndarray:
    g_low, g_high = row["green_low"], row["green_high"]
    r_low = row["red_low"]
    r_high_eff = effective_red_high(row, open_ended_red_threshold)

    center = (g_low + g_high) / 2.0
    green_width = max(g_high - g_low, 1e-6)

    # Plausible support for generation extended beyond G¬R zones to allow for variability:
    lower = min(g_low, r_low) - 0.5 * green_width
    upper = max(g_high, r_high_eff) + 0.5 * green_width

    lower, upper = apply_feature_bounds(row["feature"], lower, upper)

    # Truncated normal centered in green zone:
    sd = max(green_width / 2.0, 1e-6)
    a = (lower - center) / sd
    b = (upper - center) / sd

    return truncnorm.ppf(u, a, b, loc=center, scale=sd)


def sample_structural_values(
    u: np.ndarray,
    row: pd.Series,
    open_ended_red_threshold: float,
) -> np.ndarray:
    """
    For structural features, we sample uniformly across the entire plausible range defined by the
    min/max of green and red zones.
    """
    g_low, g_high = row["green_low"], row["green_high"]
    r_low, r_high = row["red_low"], effective_red_high(row, open_ended_red_threshold)

    lower = min(g_low, g_high, r_low, r_high)
    upper = max(g_low, g_high, r_low, r_high)

    lower, upper = apply_feature_bounds(row["feature"], lower, upper)

    return uniform.ppf(u, loc=lower, scale=max(upper - lower, 1e-6))


# ============================================================
# SECTION 3.3.2 - Correlation Structure:
#     Gaussian copula with weak intra-category correlation.
#     If rho == 0 -> independence fallback.
# ============================================================

def generate_uniform_copula(
    schema: pd.DataFrame,
    n: int,
    rho: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    features = schema["feature"].tolist()
    U = pd.DataFrame(index=np.arange(n), columns=features, dtype=float)

    if rho <= 0:
        for f in features:
            U[f] = rng.uniform(size=n)
        return U

    for category, cat_df in schema.groupby("category"):
        feats = cat_df["feature"].tolist()
        k = len(feats)

        if k == 1:
            U[feats[0]] = rng.uniform(size=n)
            continue

        cov = np.full((k, k), rho)
        np.fill_diagonal(cov, 1.0)

        latent = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
        U_cat = norm.cdf(latent)

        for j, feat in enumerate(feats):
            U[feat] = U_cat[:, j]

    return U