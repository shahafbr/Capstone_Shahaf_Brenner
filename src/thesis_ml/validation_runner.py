from pathlib import Path

from src.thesis_ml.config import TEST_SIZE, TRAIN_TEST_RANDOM_STATE
from src.thesis_ml.data import DataProcessor
from src.thesis_ml.preprocess import (
    build_scaled_preprocessor,
    build_unscaled_preprocessor,
)
from src.thesis_ml.model_builders import (
    build_ols,
    build_ridge,
    build_lasso,
    build_random_forest,
    build_gradient_boosting,
)
from src.thesis_ml.results import ResultsManager
from src.thesis_ml.training import ModelTrainer

# FIXED_MODEL_PARAMS - stores the selected hyperparameters used for validation runs.
# These are the already chosen "best" settings, so validation scenarios can reuse them without running a new grid search each time.
FIXED_MODEL_PARAMS = {
    "OLS": {},
    "Ridge": {
        "model__alpha": 100.0,
    },
    "Lasso": {
        "model__alpha": 0.01,
    },
    "RandomForest": {
        "model__max_depth": None,
        "model__min_samples_leaf": 1,
        "model__min_samples_split": 5,
        "model__n_estimators": 300,
    },
    "GradientBoosting": {
        "model__learning_rate": 0.05,
        "model__max_depth": 2,
        "model__n_estimators": 300,
        "model__subsample": 0.8,
    },
}

# run_validation_models - executes the fixed-parameter validation pipeline for one robustness scenario:
def run_validation_models(data_path: str | Path, scenario_name: str):
    """
    This function:
        1. loads the scenario dataset.
        2. prepares features and target.
        3. creates the train/test split.
        4. builds the correct preprocessors for linear vs tree-based models.
        5. trains all model families using fixed hyperparameters.
        6. saves outputs into scenario-specific validation folders.
    """
    data_path = Path(data_path)
    validation_root = Path("validation_results") / scenario_name

    # Loading the dataset and splitting it into predictors and target:
    processor = DataProcessor(data_path)
    df = processor.load_data()
    X, y = processor.prepare_features_and_target(df)

    # Creating the held-out test split using the global config settings:
    X_train, X_test, y_train, y_test = processor.split_data(
        X, y, TEST_SIZE, TRAIN_TEST_RANDOM_STATE
    )

    scaled_preprocessor = build_scaled_preprocessor(X_train)        # Linear and regularized models require scaling.
    unscaled_preprocessor = build_unscaled_preprocessor(X_train)    # Tree-based models without scaling.

    # Collecting all of the model outputs in one list for logging / printing:
    results = []

    # ------------------------------------------------------------
    # Linear model family
    # ------------------------------------------------------------
    linear_results_manager = ResultsManager(validation_root / "linear")
    linear_trainer = ModelTrainer(linear_results_manager)

    # Building the OLS pipeline and apply the fixed validation parameters:
    ols_spec = build_ols(scaled_preprocessor)
    ols_pipeline = ols_spec["pipeline"]
    ols_pipeline.set_params(**FIXED_MODEL_PARAMS["OLS"])

    # Training and evaluating OLS under the current validation scenario:
    results.append(
        linear_trainer.train_fixed_model(
            model_name=ols_spec["model_name"],
            pipeline=ols_pipeline,
            best_params=FIXED_MODEL_PARAMS["OLS"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # ------------------------------------------------------------
    # Regularized linear model family
    # ------------------------------------------------------------
    regularized_results_manager = ResultsManager(validation_root / "regularized")
    regularized_trainer = ModelTrainer(regularized_results_manager)

    # Training and evaluating Ridge and Lasso using the same scaled preprocessing:
    for builder in [build_ridge, build_lasso]:
        spec = builder(scaled_preprocessor)
        model_name = spec["model_name"]
        pipeline = spec["pipeline"]
        fixed_params = FIXED_MODEL_PARAMS[model_name]
        pipeline.set_params(**fixed_params)            # Applying the fixed validation parameters to the pipeline before training.

        results.append(
            regularized_trainer.train_fixed_model(
                model_name=model_name,
                pipeline=pipeline,
                best_params=fixed_params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )

    # ------------------------------------------------------------
    # Tree-based model family
    # ------------------------------------------------------------
    tree_results_manager = ResultsManager(validation_root / "tree")
    tree_trainer = ModelTrainer(tree_results_manager)

    # Training and evaluating Random Forest and Gradient Boosting using unscaled preprocessing:
    for builder in [build_random_forest, build_gradient_boosting]:
        spec = builder(unscaled_preprocessor)
        model_name = spec["model_name"]
        pipeline = spec["pipeline"]
        fixed_params = FIXED_MODEL_PARAMS[model_name]
        pipeline.set_params(**fixed_params)            # Applying the fixed validation parameters to the pipeline before training.

        results.append(
            tree_trainer.train_fixed_model(
                model_name=model_name,
                pipeline=pipeline,
                best_params=fixed_params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )

    # Returning all of the model results for the scenario:
    return results
