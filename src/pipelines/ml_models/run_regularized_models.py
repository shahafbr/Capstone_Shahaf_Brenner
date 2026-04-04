from src.thesis_ml.config import (
    DATA_PATH, RESULTS_ROOT,
    TEST_SIZE, TRAIN_TEST_RANDOM_STATE
)
from src.thesis_ml.data import DataProcessor
from src.thesis_ml.preprocess import build_scaled_preprocessor
from src.thesis_ml.model_builders import build_ridge, build_lasso
from src.thesis_ml.results import ResultsManager
from src.thesis_ml.training import ModelTrainer


def main():
    processor = DataProcessor(DATA_PATH)
    df = processor.load_data()
    X, y = processor.prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = processor.split_data(
        X, y, TEST_SIZE, TRAIN_TEST_RANDOM_STATE
    )

    preprocessor = build_scaled_preprocessor(X_train)

    results_manager = ResultsManager(RESULTS_ROOT / "regularized")
    trainer = ModelTrainer(results_manager)

    for builder in [build_ridge, build_lasso]:
        spec = builder(preprocessor)
        result = trainer.train_model(
            spec["model_name"],
            spec["pipeline"],
            spec["param_grid"],
            X_train, y_train, X_test, y_test
        )
        print(result)


if __name__ == "__main__":
    main()