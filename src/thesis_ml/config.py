from pathlib import Path

# Path to the main synthetic dataset used for model training and evaluation:
DATA_PATH = Path(r"data\DB\synthetic_full_5000.csv")


TARGET_COL = "burnout_risk_index"

# Columns to exclude from the feature matrix before training (e.g., latent variables that won't be present in real data):
DROP_COLS = ["latent_y", "latent_y_star"]

# 20% of the dataset reserved for the held-out test set.
TEST_SIZE = 0.2

# Random seed for the train/test split to ensure reproducibility:
TRAIN_TEST_RANDOM_STATE = 42

# Number of cross-validation folds used during model evaluation / tuning:
CV_N_SPLITS = 5

# Shuffle the data before creating CV folds:
CV_SHUFFLE = True

# Random seed for cross-validation shuffling, ensuring stable fold creation across runs:
CV_RANDOM_STATE = 42

# Scoring metrics used during cross-validation and grid search:
SCORING = {
    "rmse": "neg_root_mean_squared_error", # Negated because scikit-learn expects higher scores to be better.
    "mae": "neg_mean_absolute_error", # Negated because scikit-learn expects higher scores to be better.
    "r2": "r2"
}

# Metric used to select the best model during grid search refitting:
REFIT_METRIC = "rmse"


RESULTS_ROOT = Path("results")