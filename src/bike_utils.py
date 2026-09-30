"""Shared helpers for the Bike Price Prediction labs (regression).

Everything that more than one lab script needs lives here so the notebook logic
(Mani.ipynb) is defined exactly once: paths, feature lists, cleaning, feature
engineering, the preprocessing ColumnTransformer, metrics and model configs.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Constants (taken from the notebook)
# ----------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/BIKE DETAILS.csv"
TARGET = "selling_price"
REFERENCE_YEAR = 2026
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["name", "seller_type", "owner"]
NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "ex_showroom_price",
    "bike_age",
    "km_per_year",
    "log_km_driven",
    "log_ex_showroom",
]

EXPERIMENT_NAME = "Bike_Price_Prediction"
REGISTERED_MODEL_NAME = "Bike_Price_Production_Model"


# ----------------------------------------------------------------------------
# Cleaning / feature engineering
# ----------------------------------------------------------------------------
def load_raw(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean_and_engineer(df):
    """Row-wise steps from the notebook that do NOT learn anything from data,
    so they are safe to run before the train/test split.

    duplicates -> numeric coercion -> drop rows without target -> fill text
    columns -> bike_age, km_per_year, log_km_driven
    """
    df = df.drop_duplicates().copy()

    for col in ["year", "km_driven", "ex_showroom_price", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")

    df["bike_age"] = REFERENCE_YEAR - df["year"]
    df["km_per_year"] = df["km_driven"] / (df["bike_age"] + 1)
    df["log_km_driven"] = np.log1p(df["km_driven"])
    return df


def impute_ex_showroom_notebook_style(df):
    """Exactly what the notebook does: group-wise median by bike name, then a
    global median, computed on the FULL dataset (before the split).
    Used by Lab 4 so the tracked runs match the notebook's numbers.
    """
    df = df.copy()
    df["ex_showroom_price"] = df.groupby("name")["ex_showroom_price"].transform(
        lambda x: x.fillna(x.median())
    )
    df["ex_showroom_price"] = df["ex_showroom_price"].fillna(df["ex_showroom_price"].median())
    df["log_ex_showroom"] = np.log1p(df["ex_showroom_price"])
    return df


class ExShowroomImputer(BaseEstimator, TransformerMixin):
    """Leakage-safe version of the notebook's ex-showroom imputation (Lab 5).

    fit() learns per-bike-name medians and a global median from the TRAINING
    rows only; transform() applies them to any data and adds log_ex_showroom.
    """

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby("name")["ex_showroom_price"].median().dropna()
        self.global_median_ = float(X["ex_showroom_price"].median())
        return self

    def transform(self, X):
        X = X.copy()
        fill = X["name"].map(self.group_medians_)
        X["ex_showroom_price"] = X["ex_showroom_price"].fillna(fill).fillna(self.global_median_)
        X["log_ex_showroom"] = np.log1p(X["ex_showroom_price"])
        return X


def build_column_transformer():
    """The notebook's preprocessor (median for numbers, most-frequent + one-hot
    for text). Dense output so the result can be stored as .npy files."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_production_preprocessor():
    """Lab 5: imputer + ColumnTransformer as one fitted-on-train-only object."""
    return Pipeline(
        steps=[
            ("ex_showroom_imputer", ExShowroomImputer()),
            ("column_transformer", build_column_transformer()),
        ]
    )


# ----------------------------------------------------------------------------
# Metrics and models
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def get_model_configs():
    """The five models compared in the notebook (same hyper-parameters), plus
    a deeper XGBoost variant used as a 'challenger' in Lab 6.
    Returns {run_name: (model_family, estimator_class, params)}."""
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    return {
        "LinearRegression": ("LinearRegression", LinearRegression, {}),
        "RandomForest": (
            "RandomForest",
            RandomForestRegressor,
            dict(n_estimators=500, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=0.8, random_state=42, n_jobs=-1),
        ),
        "GradientBoosting": (
            "GradientBoosting",
            GradientBoostingRegressor,
            dict(n_estimators=500, learning_rate=0.03, max_depth=4, min_samples_split=2,
                 min_samples_leaf=1, loss="huber", random_state=42),
        ),
        "ExtraTrees": (
            "ExtraTrees",
            ExtraTreesRegressor,
            dict(n_estimators=500, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=0.8, random_state=42, n_jobs=-1),
        ),
        "XGBoost": (
            "XGBoost",
            XGBRegressor,
            dict(n_estimators=1000, learning_rate=0.03, max_depth=5, min_child_weight=2,
                 subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1,
                 objective="reg:squarederror", random_state=42, n_jobs=-1),
        ),
        "XGBoost_Deep": (
            "XGBoost",
            XGBRegressor,
            dict(n_estimators=1500, learning_rate=0.02, max_depth=6, min_child_weight=2,
                 subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1,
                 objective="reg:squarederror", random_state=42, n_jobs=-1),
        ),
    }
