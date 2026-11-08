"""
car_utils.py
============
Shared, reusable code for the Indian Pre-Owned Car Price Prediction project.

Both the Jupyter notebook (training) and the Streamlit app (inference) import
this module, so the *exact same* cleaning + feature engineering is applied
at training time and at prediction time (no train/serve skew).

Contents
--------
load_data()        : read a CSV and normalise column names (e.g. 'Distance ' -> 'Distance')
CarPreprocessor    : scikit-learn compatible transformer (cleaning, imputation,
                     feature engineering, encoding)
regression_report(): RMSE / MAE / R2 helper
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CURRENT_YEAR = 2025                 # dataset reference year (manufacture_year + Age = 2025)
OWNER_MAP = {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4}

# Physical plausibility limits used to flag corrupted values as "missing"
MAX_DISPLACEMENT_CC = 6500          # e.g. 32000 cc is a data-entry error
MIN_DISPLACEMENT_CC = 600
MAX_POWER = 500
MIN_POWER = 20
MAX_DISTANCE_KM = 500_000           # e.g. 9,899,800 km is impossible

RAW_FEATURES = [
    "Maker", "model", "Location", "Distance", "Owner Type", "manufacture_year",
    "Age of car", "engine_displacement", "engine_power", "body_type",
    "Vroom Audit Rating", "transmission", "door_count", "seat_count", "fuel_type",
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from column names (the raw file has 'Distance ')."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def load_data(path_or_buffer) -> pd.DataFrame:
    """Read a CSV (path or uploaded file) and tidy the column names."""
    return standardise_columns(pd.read_csv(path_or_buffer))


# --------------------------------------------------------------------------- #
# Preprocessing transformer
# --------------------------------------------------------------------------- #
class CarPreprocessor(BaseEstimator, TransformerMixin):
    """Clean, impute, engineer and encode the raw car dataframe.

    Fitted statistics (medians / modes / category lists) are learned ONLY from
    the training fold, so cross-validation is leakage-free.

    Steps
    -----
    1. Invalid-value handling  : impossible numbers -> NaN
       (displacement > 6500 cc, distance > 500,000 km, zero km on an old car ...)
    2. Imputation              : group-wise medians / modes
         Distance     -> median by Age of car
         displacement -> median by (maker, model, fuel)
         engine_power -> median by (maker, model, displacement)  -> fallback (maker, model)
         door/seat    -> mode by (maker, model)
         body_type    -> 'standard' (blank in the raw data means a regular body)
    3. Feature engineering     : km_per_year, log_distance, power_per_litre,
                                 missing-value flags, ordinal owner, binary
                                 transmission / fuel
    4. Encoding                : one-hot for maker, maker_model, location
    """

    def fit(self, X: pd.DataFrame, y=None):
        df = self._basic_clean(X)

        # ---- imputation statistics (learned on training data only) ----
        self.global_median_ = {
            c: float(df[c].median())
            for c in ["Distance", "engine_displacement", "engine_power"]
        }
        self.dist_by_age_ = df.groupby("Age of car")["Distance"].median().to_dict()
        self.disp_by_mfuel_ = df.groupby(["maker_model", "fuel_type"])["engine_displacement"].median().to_dict()
        self.disp_by_model_ = df.groupby("maker_model")["engine_displacement"].median().to_dict()
        self.power_by_md_ = df.groupby(["maker_model", "engine_displacement"])["engine_power"].median().to_dict()
        self.power_by_model_ = df.groupby("maker_model")["engine_power"].median().to_dict()
        self.door_by_model_ = df.groupby("maker_model")["door_count"].agg(_mode).to_dict()
        self.seat_by_model_ = df.groupby("maker_model")["seat_count"].agg(_mode).to_dict()

        # ---- category lists for stable one-hot columns ----
        self.makers_ = sorted(df["Maker"].dropna().unique())
        self.maker_models_ = sorted(df["maker_model"].dropna().unique())
        self.locations_ = sorted(df["Location"].dropna().unique())

        self.feature_names_ = list(self._build(df).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build(self._basic_clean(X))

    # -- kept so the object works nicely inside sklearn Pipelines --
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _basic_clean(X: pd.DataFrame) -> pd.DataFrame:
        """Column-name tidy, type coercion and invalid-value -> NaN (no statistics needed)."""
        df = standardise_columns(X)
        for c in ["Maker", "model", "Location", "Owner Type", "transmission", "fuel_type", "body_type"]:
            if c in df:
                df[c] = df[c].astype("string").str.strip().str.lower() if c not in ("Location", "Owner Type") \
                    else df[c].astype("string").str.strip()
        for c in ["Distance", "engine_displacement", "engine_power", "door_count",
                  "seat_count", "Age of car", "Vroom Audit Rating", "manufacture_year"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["maker_model"] = df["Maker"].astype(str) + "_" + df["model"].astype(str)

        # --- structural fixes: impossible values become missing ---
        bad_disp = (df["engine_displacement"] > MAX_DISPLACEMENT_CC) | (df["engine_displacement"] < MIN_DISPLACEMENT_CC)
        df["disp_was_invalid"] = bad_disp.astype(int)
        df.loc[bad_disp, "engine_displacement"] = np.nan

        bad_pow = (df["engine_power"] > MAX_POWER) | (df["engine_power"] < MIN_POWER)
        df.loc[bad_pow, "engine_power"] = np.nan

        bad_dist = (df["Distance"] > MAX_DISTANCE_KM) | (df["Distance"] < 0) | \
                   ((df["Distance"] == 0) & (df["Age of car"] >= 2))   # 0 km on a 2+ year old car
        df.loc[bad_dist, "Distance"] = np.nan

        # Age is the reliable field (manufacture_year is redundant & sometimes missing)
        df["Age of car"] = df["Age of car"].clip(lower=0)
        return df

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # ---- missing-value indicator flags (missingness itself can be informative) ----
        out["distance_missing"] = df["Distance"].isna().astype(int)
        out["power_missing"] = df["engine_power"].isna().astype(int)
        out["disp_invalid_or_missing"] = (df["engine_displacement"].isna() | (df["disp_was_invalid"] == 1)).astype(int)
        out["has_body_type"] = df["body_type"].notna().astype(int)

        # ---- imputation ----
        age = df["Age of car"]
        dist = df["Distance"].copy()
        dist = dist.fillna(age.map(self.dist_by_age_)).fillna(self.global_median_["Distance"])

        keys = list(zip(df["maker_model"], df["fuel_type"]))
        disp = df["engine_displacement"].copy()
        disp = disp.fillna(pd.Series([self.disp_by_mfuel_.get(k) for k in keys], index=df.index))
        disp = disp.fillna(df["maker_model"].map(self.disp_by_model_)).fillna(self.global_median_["engine_displacement"])

        pkeys = list(zip(df["maker_model"], disp))
        power = df["engine_power"].copy()
        power = power.fillna(pd.Series([self.power_by_md_.get(k) for k in pkeys], index=df.index))
        power = power.fillna(df["maker_model"].map(self.power_by_model_)).fillna(self.global_median_["engine_power"])

        doors = df["door_count"].fillna(df["maker_model"].map(self.door_by_model_)).fillna(4)
        seats = df["seat_count"].fillna(df["maker_model"].map(self.seat_by_model_)).fillna(5)

        # ---- numeric features ----
        out["age"] = age
        out["distance"] = dist
        out["log_distance"] = np.log1p(dist)
        out["km_per_year"] = dist / np.maximum(age, 1)
        out["engine_displacement"] = disp
        out["engine_power"] = power
        out["power_per_litre"] = power / (disp / 1000.0)
        out["door_count"] = doors
        out["seat_count"] = seats
        out["audit_rating"] = df["Vroom Audit Rating"]
        out["owner_ordinal"] = df["Owner Type"].map(OWNER_MAP)
        out["is_automatic"] = df["transmission"].eq("auto").fillna(False).astype(int)
        out["is_diesel"] = df["fuel_type"].eq("diesel").fillna(False).astype(int)

        # ---- one-hot categoricals (fixed category lists => stable columns) ----
        for prefix, col, cats in [("maker", "Maker", self.makers_),
                                  ("mm", "maker_model", self.maker_models_),
                                  ("loc", "Location", self.locations_)]:
            for c in cats:
                out[f"{prefix}_{c}"] = df[col].eq(c).fillna(False).astype(int)

        # Plain float64 everywhere: newer pandas versions can return pyarrow-backed integer
        # columns for the mapped / string-derived features, which LightGBM and XGBoost reject.
        return out.astype("float64")


def _mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


# --------------------------------------------------------------------------- #
# Evaluation helper
# --------------------------------------------------------------------------- #
def regression_report(y_true, y_pred) -> dict:
    """RMSE, MAE and R2 on the original (rupee) scale."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
