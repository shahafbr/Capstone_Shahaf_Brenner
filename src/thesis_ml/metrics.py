from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# This module contains utility functions for computing regression metrics used in model evaluation and final analysis.
def compute_regression_metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }