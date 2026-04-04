from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SCHEMA_PATH = Path(r"data\Expert_Aggregated_DB\aggregated_feature_schema.csv")
OUTPUT_ROOT = Path(r"data\DB")

SMALL_N = 1000
FULL_N = 5000
RANDOM_SEED = 42

# Weak intra-category correlation strength, 0.0 for fallback independent configuration:
INTRA_CATEGORY_RHO = 0.15

# Noise standard deviation in latent y* space before multiplying by 100:
NOISE_STD = 0.05

# cap the effective red upper bound for generation when values exceed this threshold:
OPEN_ENDED_RED_THRESHOLD = 1500.0


# 1) Feature type - Section 2.3.1 sampling strategy:
FEATURE_TYPE = {
    "Night_work": "behavioral",
    "After_hours_communication": "behavioral",
    "Weekend_activity": "behavioral",
    "Backlog_size": "structural",
    "Long_days": "behavioral",
    "Reopened_tasks": "behavioral",
    "Zombie_logged_in_state": "behavioral",
    "Context_switching": "behavioral",
    "Meeting_density": "behavioral",
    "Error_bug_rate": "behavioral",
    "Sick_leave_frequency": "behavioral",
    "Work_time_VS_clock_in_out": "behavioral",
    "Break_frequency": "behavioral",
    "Reactive_meeting_load": "behavioral",
    "PTO_usage": "structural",
    "Irregularity": "behavioral",
    "Task_completion": "behavioral",
    "Notification_load": "behavioral",
    "Communication_variability": "behavioral",
    "Meetings_missed": "behavioral",
}

# 2) Direction - Section 3.2.3 / 3.3.4:
DIRECTION = {
    "Night_work": "increasing",
    "After_hours_communication": "increasing",
    "Weekend_activity": "increasing",
    "Backlog_size": "increasing",
    "Long_days": "increasing",
    "Reopened_tasks": "increasing",
    "Zombie_logged_in_state": "increasing",
    "Context_switching": "increasing",
    "Meeting_density": "increasing",
    "Error_bug_rate": "increasing",
    "Sick_leave_frequency": "increasing",
    "Work_time_VS_clock_in_out": "decreasing",
    "Break_frequency": "increasing",
    "Reactive_meeting_load": "increasing",
    "PTO_usage": "decreasing",
    "Irregularity": "increasing",
    "Task_completion": "increasing",
    "Notification_load": "increasing",
    "Communication_variability": "u_shaped",
    "Meetings_missed": "increasing",
}

# 3) Category - required for weak intra-category correlation:
CATEGORY = {
    "Night_work": "Temporal",
    "After_hours_communication": "Communication",
    "Weekend_activity": "Temporal",
    "Backlog_size": "Workload",
    "Long_days": "Temporal",
    "Reopened_tasks": "Workload",
    "Zombie_logged_in_state": "Passive",
    "Context_switching": "Focus",
    "Meeting_density": "Meetings",
    "Error_bug_rate": "Performance",
    "Sick_leave_frequency": "Attendance",
    "Work_time_VS_clock_in_out": "Temporal",
    "Break_frequency": "Focus",
    "Reactive_meeting_load": "Meetings",
    "PTO_usage": "Attendance",
    "Irregularity": "Temporal",
    "Task_completion": "Performance",
    "Notification_load": "Focus",
    "Communication_variability": "Communication",
    "Meetings_missed": "Meetings",
}

# 4) Non-negativity constraint for certain features (enforced in sampling):
NON_NEGATIVE_FEATURES = {
    "Night_work": True,
    "After_hours_communication": True,
    "Weekend_activity": True,
    "Backlog_size": True,
    "Long_days": True,
    "Reopened_tasks": True,
    "Zombie_logged_in_state": True,
    "Context_switching": True,
    "Meeting_density": True,
    "Error_bug_rate": True,
    "Sick_leave_frequency": True,
    "Work_time_VS_clock_in_out": False,
    "Break_frequency": True,
    "Reactive_meeting_load": True,
    "PTO_usage": True,
    "Irregularity": True,
    "Task_completion": True,
    "Notification_load": True,
    "Communication_variability": False,
    "Meetings_missed": True,
}

# Automatically creates:
#  1) __init__() with all fields as parameters.
#  2) a readable object representation for debugging.
#  3) field defaults in one compact place.
#  4) correct handling of feature_drop through field(default_factory=list).
@dataclass
class SyntheticScenarioConfig:
    # scenario_name - controls the output subfolder so each circularity test writes to a separate DB:
    scenario_name: str = "baseline"

    # schema_path - stays configurable so future tests can point to alternate schemas:
    schema_path: Path = SCHEMA_PATH

    output_root: Path = OUTPUT_ROOT

    # separate seeds to vary observation sampling and weight uncertainty independently:
    data_seed_small: int = RANDOM_SEED
    data_seed_full: int = RANDOM_SEED + 1
    weight_seed_small: int | None = None
    weight_seed_full: int | None = None

    # keeping both DB sizes inside the same scenario object for pipeline simplicity:
    small_n: int = SMALL_N
    full_n: int = FULL_N

    # circularity / robustness controls -> scenario-level parameters:
    intra_category_rho: float = INTRA_CATEGORY_RHO
    noise_std: float = NOISE_STD
    open_ended_red_threshold: float = OPEN_ENDED_RED_THRESHOLD

    # Allowing direct alpha overrides for alpha sensitivity tests without editing the schema:
    alpha_override: float | None = None

    # feature_drop - removes variables from generation and target construction for ablation tests.
    feature_drop: list[str] = field(default_factory=list)

    # include_latent_columns - allows to keep/hide latent columns depending on the test:
    include_latent_columns: bool = True

    # metadata_filename - keeps the save convention explicit and reusable across scenarios:
    metadata_filename: str = "scenario_metadata.json"