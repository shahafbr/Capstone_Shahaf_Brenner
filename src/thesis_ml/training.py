import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from src.thesis_ml.config import (
    CV_N_SPLITS, CV_SHUFFLE, CV_RANDOM_STATE,
    SCORING, REFIT_METRIC,
    TEST_SIZE, TRAIN_TEST_RANDOM_STATE
)
from src.thesis_ml.metrics import compute_regression_metrics


class ModelTrainer:
    # results_manager - utility object responsible for creating run folders and saving outputs such as metrics, predictions, configs, and summaries.
    def __init__(self, results_manager):
        self.results_manager = results_manager

    # _build_fold_indices - creates and stores the exact CV split indices used for training.
    # Saving these indices improves reproducibility and allows later inspection of the fold structure.
    def _build_fold_indices(self, X_train):
        kf = KFold(
            n_splits=CV_N_SPLITS,
            shuffle=CV_SHUFFLE,
            random_state=CV_RANDOM_STATE
        )

        folds = []
        for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
            folds.append({
                "fold": fold_id,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            })
        return folds

    # train_model - performs full model training with GridSearchCV:
    def train_model(self, model_name, pipeline, param_grid, X_train, y_train, X_test, y_test):
        """
        1. Creates a unique run directory for the model.
        2. saves CV fold indices.
        3. runs hyperparameter search.
        4. evaluates the best model on the held-out test set.
        5. saves CV results, config, predictions, metrics, and feature importance.
        """
        run_id, run_dir = self.results_manager.create_run_dir(model_name)

        # Saving the exact fold structure used in CV for transparency and reproducibility:
        folds = self._build_fold_indices(X_train)
        self.results_manager.save_json(
            {
                "cv_n_splits": CV_N_SPLITS,
                "cv_shuffle": CV_SHUFFLE,
                "cv_random_state": CV_RANDOM_STATE,
                "folds": folds
            },
            run_dir / "cv_fold_indices.json"
        )

        # GridSearchCV - tunes hyperparameters using the configured scoring metrics.
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid if param_grid else {},
            scoring=SCORING,
            refit=REFIT_METRIC,  # means the best model is selected according to that metric.
            cv=KFold(
                n_splits=CV_N_SPLITS,
                shuffle=CV_SHUFFLE,
                random_state=CV_RANDOM_STATE
            ),
            n_jobs=-1,
            return_train_score=True
        )

        # Fitting the hyperparameter search on the training set only. This ensures that the test set remains completely unseen until the final evaluation step:
        grid.fit(X_train, y_train)

        # Retrieving the best fitted model and generate held-out test predictions:
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Computing the final regression metrics on the held-out test set:
        test_metrics = compute_regression_metrics(y_test, y_pred)

        # Saving the full GridSearchCV results table for later analysis:
        cv_results_df = pd.DataFrame(grid.cv_results_)
        self.results_manager.save_dataframe(cv_results_df, run_dir / "cv_results.csv")

        # Saving the run metadata and selected hyperparameters:
        self.results_manager.save_json(
            {
                "model_name": model_name,
                "param_grid": param_grid,
                "best_params": grid.best_params_,
                "refit_metric": REFIT_METRIC,
                "scoring": list(SCORING.keys()),
                "test_size": TEST_SIZE,
                "train_test_random_state": TRAIN_TEST_RANDOM_STATE
            },
            run_dir / "run_config.json"
        )

        # Saving row-level held-out test predictions:
        preds_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred
        })
        self.results_manager.save_dataframe(preds_df, run_dir / "test_predictions.csv")

        # Saving the final held-out test metrics as JSON:
        self.results_manager.save_json(test_metrics, run_dir / "test_metrics.json")
        
        # Recovering the fitted model and preprocessor steps from the best pipeline:
        model_step = best_model.named_steps["model"]
        preprocessor = best_model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        # Saving the feature importance for linear models using coefficients (Sorting by absolute value makes coefficient magnitude easier to compare):
        if hasattr(model_step, "coef_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model_step.coef_
            }).sort_values("importance", key=np.abs, ascending=False)
            self.results_manager.save_dataframe(importance_df, run_dir / "feature_importance.csv")

        # Saving the feature importance for tree-based models using feature_importances_:
        elif hasattr(model_step, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model_step.feature_importances_
            }).sort_values("importance", ascending=False)
            self.results_manager.save_dataframe(importance_df, run_dir / "feature_importance.csv")

        # Appending one summary row to the cumulative runs_summary.csv file:
        self.results_manager.append_summary_row({
            "run_id": run_id,
            "model_name": model_name,
            "cv_n_splits": CV_N_SPLITS,
            "cv_shuffle": CV_SHUFFLE,
            "cv_random_state": CV_RANDOM_STATE,
            "train_test_random_state": TRAIN_TEST_RANDOM_STATE,
            "test_size": TEST_SIZE,
            "best_params": json.dumps(grid.best_params_),
            "best_cv_rmse": float(-grid.best_score_),
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"]
        })

        # Returning a compact summary for immediate pipeline logging / printing:
        return {
            "run_id": run_id,
            "best_params": grid.best_params_,
            "test_metrics": test_metrics
        }
    
    # train_fixed_model - trains and evaluates a model using already selected hyperparameters:
    def train_fixed_model(self, model_name, pipeline, best_params, X_train, y_train, X_test, y_test):
        run_id, run_dir = self.results_manager.create_run_dir(model_name)

        # Saving the exact fold structure used for this fixed-parameter run:
        folds = self._build_fold_indices(X_train)
        self.results_manager.save_json(
            {
                "cv_n_splits": CV_N_SPLITS,
                "cv_shuffle": CV_SHUFFLE,
                "cv_random_state": CV_RANDOM_STATE,
                "folds": folds
            },
            run_dir / "cv_fold_indices.json"
        )

        # Fitting one fixed model specification per fold using the already selected hyperparameters.
        # This recreates fold-level CV metrics in a way that is comparable to GridSearchCV output.
        cv_records = []
        kf = KFold(
            n_splits=CV_N_SPLITS,
            shuffle=CV_SHUFFLE,
            random_state=CV_RANDOM_STATE
        )

        for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            # Fitting on the fold's training portion:
            pipeline.fit(X_tr, y_tr)

            # Generating both training and validation predictions for diagnostics:
            y_tr_pred = pipeline.predict(X_tr)
            y_val_pred = pipeline.predict(X_val)

            # Computing the metrics separately for train and validation portions:
            train_metrics = compute_regression_metrics(y_tr, y_tr_pred)
            val_metrics = compute_regression_metrics(y_val, y_val_pred)

            # Storing fold-level metrics in a structure similar to sklearn cv_results_ (Validation RMSE and MAE are negated to match sklearn's scoring convention):
            cv_records.append({
                "fold": fold_id,
                "split{}_train_rmse".format(fold_id - 1): train_metrics["rmse"],
                "split{}_train_mae".format(fold_id - 1): train_metrics["mae"],
                "split{}_train_r2".format(fold_id - 1): train_metrics["r2"],
                "split{}_test_rmse".format(fold_id - 1): -val_metrics["rmse"],
                "split{}_test_mae".format(fold_id - 1): -val_metrics["mae"],
                "split{}_test_r2".format(fold_id - 1): val_metrics["r2"],
            })

        # Converting the fold records into a single-row cv_results table that resembles GridSearchCV output:
        cv_row = {
            "params": json.dumps(best_params),
            "rank_test_rmse": 1,
            "rank_test_mae": 1,
            "rank_test_r2": 1,
        }

        # Flattening all of the fold metrics into one output row:
        for i in range(CV_N_SPLITS):
            fold_record = cv_records[i]
            cv_row[f"split{i}_train_rmse"] = fold_record[f"split{i}_train_rmse"]
            cv_row[f"split{i}_train_mae"] = fold_record[f"split{i}_train_mae"]
            cv_row[f"split{i}_train_r2"] = fold_record[f"split{i}_train_r2"]
            cv_row[f"split{i}_test_rmse"] = fold_record[f"split{i}_test_rmse"]
            cv_row[f"split{i}_test_mae"] = fold_record[f"split{i}_test_mae"]
            cv_row[f"split{i}_test_r2"] = fold_record[f"split{i}_test_r2"]

        # Collecting the metric lists so mean/std can be computed across folds:
        test_rmse_vals = [cv_row[f"split{i}_test_rmse"] for i in range(CV_N_SPLITS)]
        test_mae_vals = [cv_row[f"split{i}_test_mae"] for i in range(CV_N_SPLITS)]
        test_r2_vals = [cv_row[f"split{i}_test_r2"] for i in range(CV_N_SPLITS)]

        train_rmse_vals = [cv_row[f"split{i}_train_rmse"] for i in range(CV_N_SPLITS)]
        train_mae_vals = [cv_row[f"split{i}_train_mae"] for i in range(CV_N_SPLITS)]
        train_r2_vals = [cv_row[f"split{i}_train_r2"] for i in range(CV_N_SPLITS)]

        # Computing the fold-mean and fold-std summaries for train and validation metrics:
        cv_row["mean_test_rmse"] = float(np.mean(test_rmse_vals))
        cv_row["std_test_rmse"] = float(np.std(test_rmse_vals, ddof=0))
        cv_row["mean_test_mae"] = float(np.mean(test_mae_vals))
        cv_row["std_test_mae"] = float(np.std(test_mae_vals, ddof=0))
        cv_row["mean_test_r2"] = float(np.mean(test_r2_vals))
        cv_row["std_test_r2"] = float(np.std(test_r2_vals, ddof=0))

        cv_row["mean_train_rmse"] = float(np.mean(train_rmse_vals))
        cv_row["std_train_rmse"] = float(np.std(train_rmse_vals, ddof=0))
        cv_row["mean_train_mae"] = float(np.mean(train_mae_vals))
        cv_row["std_train_mae"] = float(np.std(train_mae_vals, ddof=0))
        cv_row["mean_train_r2"] = float(np.mean(train_r2_vals))
        cv_row["std_train_r2"] = float(np.std(train_r2_vals, ddof=0))

        # Saving the synthetic cv_results table for later analysis:
        cv_results_df = pd.DataFrame([cv_row])
        self.results_manager.save_dataframe(cv_results_df, run_dir / "cv_results.csv")

        # final fit on full training set:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Computing held-out test metrics:
        test_metrics = compute_regression_metrics(y_test, y_pred)

        # Saving run configuration, explicitly marking that no grid search was used here:
        self.results_manager.save_json(
            {
                "model_name": model_name,
                "param_grid": {},
                "best_params": best_params,
                "refit_metric": REFIT_METRIC,
                "scoring": list(SCORING.keys()),
                "test_size": TEST_SIZE,
                "train_test_random_state": TRAIN_TEST_RANDOM_STATE,
                "selection_mode": "fixed_hyperparameters"
            },
            run_dir / "run_config.json"
        )

        # Saving the row-level held-out test predictions:
        preds_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred
        })
        self.results_manager.save_dataframe(preds_df, run_dir / "test_predictions.csv")

        # Saving final held-out test metrics:
        self.results_manager.save_json(test_metrics, run_dir / "test_metrics.json")

        # Recovering the model and preprocessor steps after the final fit:
        model_step = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        # Saving the feature importance for linear models:
        if hasattr(model_step, "coef_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model_step.coef_
            }).sort_values("importance", key=np.abs, ascending=False)
            self.results_manager.save_dataframe(importance_df, run_dir / "feature_importance.csv")

        # Saving the feature importance for tree-based models:
        elif hasattr(model_step, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model_step.feature_importances_
            }).sort_values("importance", ascending=False)
            self.results_manager.save_dataframe(importance_df, run_dir / "feature_importance.csv")

        # Appending one summary row to the cumulative runs summary file:
        self.results_manager.append_summary_row({
            "run_id": run_id,
            "model_name": model_name,
            "cv_n_splits": CV_N_SPLITS,
            "cv_shuffle": CV_SHUFFLE,
            "cv_random_state": CV_RANDOM_STATE,
            "train_test_random_state": TRAIN_TEST_RANDOM_STATE,
            "test_size": TEST_SIZE,
            "best_params": json.dumps(best_params),
            "best_cv_rmse": float(-cv_row["mean_test_rmse"]),
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"]
        })

        # Returning a compact run summary for direct logging:
        return {
            "run_id": run_id,
            "best_params": best_params,
            "test_metrics": test_metrics
        }