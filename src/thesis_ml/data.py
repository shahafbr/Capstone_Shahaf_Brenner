import pandas as pd
from sklearn.model_selection import train_test_split

from src.thesis_ml.config import TARGET_COL, DROP_COLS


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    # Load the dataset from the specified CSV file path:
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)
    
    # Prepare the feature matrix (X) and target vector (y) by separating the target column and dropping any specified columns:
    def prepare_features_and_target(self, df: pd.DataFrame):
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        X = df.drop(columns=[TARGET_COL] + cols_to_drop)
        y = df[TARGET_COL]
        return X, y

    # Split the data into training and testing sets using scikit-learn's train_test_split function, with specified test size and random state for reproducibility:
    def split_data(self, X, y, test_size, random_state):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)