"""
model1_predict.py
------------------
Loads the trained Model 1 XGBoost booster (models/model1_body_temp.json)
and runs prediction on a 36-feature DataFrame built by model1_features.py.

The booster is loaded once at import time and cached -- reloading a
3000-tree model per request would be needlessly slow.
"""

import os
import xgboost as xgb
import pandas as pd

from model1_features import FEATURE_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model1_body_temp.json")

_booster = None


def get_model() -> xgb.Booster:
    global _booster
    if _booster is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Place model1_body_temp.json in the models/ folder."
            )
        _booster = xgb.Booster()
        _booster.load_model(MODEL_PATH)
    return _booster


def predict_body_temp(features_df: pd.DataFrame) -> pd.Series:
    """
    features_df: DataFrame with all 36 FEATURE_COLUMNS (as produced by
    model1_features.build_model1_features). Column order is enforced
    here to match the model's training order exactly, regardless of the
    order columns happen to be in on the DataFrame passed in.
    Returns a pandas Series of predicted body_temperature_C, one per row.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"predict_body_temp: missing required feature columns: {missing}")

    ordered = features_df[FEATURE_COLUMNS]
    dmatrix = xgb.DMatrix(ordered, feature_names=FEATURE_COLUMNS)

    booster = get_model()
    preds = booster.predict(dmatrix)
    return pd.Series(preds, index=features_df.index, name="predicted_body_temp_C")
