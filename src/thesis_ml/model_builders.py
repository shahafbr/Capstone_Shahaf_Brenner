from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# build_ols - creates the baseline Ordinary Least Squares regression pipeline.
# This model has no hyperparameters to tune in the current setup, so the param_grid is empty.
def build_ols(preprocessor):
    return {
        "model_name": "OLS",
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "param_grid": {}
    }

# build_ridge - creates the Ridge regression pipeline.
def build_ridge(preprocessor):
    return {
        "model_name": "Ridge",
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge())
        ]),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

# build_lasso - creates the Lasso regression pipeline.
def build_lasso(preprocessor):
    return {
        "model_name": "Lasso",
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Lasso(max_iter=10000))
        ]),
        "param_grid": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
        }
    }

# build_random_forest - creates the Random Forest regression pipeline.
def build_random_forest(preprocessor):
    return {
        "model_name": "RandomForest",
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42))
        ]),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    }

# build_gradient_boosting - creates the Gradient Boosting regression pipeline.
def build_gradient_boosting(preprocessor):
    return {
        "model_name": "GradientBoosting",
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42))
        ]),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0]
        }
    }