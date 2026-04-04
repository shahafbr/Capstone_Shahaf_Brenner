from __future__ import annotations

import numpy as np
import pandas as pd

from src.synthetic_db.synthetic_sampling import effective_red_high


# ============================================================
# SECTION 3.3.4 / 3.2.3 - Risk Intensity z_j in [0,1]:
# ============================================================

def compute_z_increasing(x: np.ndarray, row: pd.Series) -> np.ndarray:
    g_high = row["green_high"]
    r_low = row["red_low"]

    lo = min(g_high, r_low)
    hi = max(g_high, r_low)

    # If G&R zones are effectively the same, we treat values above that threshold as risky:
    if np.isclose(lo, hi):
        return (x >= hi).astype(float)

    # Linear interpolation between green and red zones, clipped to [0,1]:
    z = (x - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)


def compute_z_decreasing(x: np.ndarray, row: pd.Series) -> np.ndarray:
    # Inverse of increasing relationship:
    g_low = row["green_low"]
    r_high = row["red_high"]

    lo = min(r_high, g_low)
    hi = max(r_high, g_low)

    if np.isclose(lo, hi):
        return (x <= lo).astype(float)

    z = (hi - x) / (hi - lo)
    return np.clip(z, 0.0, 1.0)


def compute_z_ushaped(
    x: np.ndarray,
    row: pd.Series,
    open_ended_red_threshold: float,
) -> np.ndarray:
    g_low, g_high = row["green_low"], row["green_high"]
    r_low, r_high = row["red_low"], effective_red_high(row, open_ended_red_threshold)

    midpoint = (g_low + g_high) / 2.0
    half_green = max((g_high - g_low) / 2.0, 1e-6)

    # Distance from midpoint at which risk should saturate:
    risky_distance = max(abs(r_low - midpoint), abs(r_high - midpoint), half_green + 1e-6)

    dist = np.abs(x - midpoint)
    z = (dist - half_green) / (risky_distance - half_green)
    return np.clip(z, 0.0, 1.0)


def compute_risk_intensity(
    x: np.ndarray,
    row: pd.Series,
    open_ended_red_threshold: float,
) -> np.ndarray:
    direction = row["direction"]
    if direction == "increasing":
        return compute_z_increasing(x, row)
    elif direction == "decreasing":
        return compute_z_decreasing(x, row)
    elif direction == "u_shaped":
        return compute_z_ushaped(x, row, open_ended_red_threshold)
    else:
        raise ValueError(f"Unknown direction '{direction}' for feature {row['feature']}")