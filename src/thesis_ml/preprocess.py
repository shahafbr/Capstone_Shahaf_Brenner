from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# build_scaled_preprocessor - creates a preprocessing pipeline for models that benefit from standardized input features (e.g., OLS, Ridge, Lasso):
def build_scaled_preprocessor(X):
    """
    The function assumes that all columns in X are numeric features, It applies:
        1. median imputation for missing values
        2. standardization to zero mean and unit variance
    """

    # Treating all columns in X as numeric predictors:
    numeric_features = X.columns.tolist()

    # Preprocessing steps applied to numeric features (replacing missing numeric values with the column median and then standardizing):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # The ColumnTransformer applies the numeric_transformer to all numeric features, and drops any other columns (if present):
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )

# build_unscaled_preprocessor - creates a preprocessing pipeline for models that do not require feature scaling (e.g., tree-based models):
def build_unscaled_preprocessor(X):
    """
    The function assumes that all columns in X are numeric features, It applies:
        1. median imputation for missing values
        2. no scaling
    """

    numeric_features = X.columns.tolist()

    # Replacing missing numeric values with the column median:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    return ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )