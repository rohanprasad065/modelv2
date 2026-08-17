"""
estrus_predict.py
------------------
Loads the two trained estrus XGBoost boosters and runs prediction on the
43-column DataFrame produced by base_12col + predicted_body_temp_C +
model2_features.build_model2_features() (see main.py steps 1-5).

Two separate models, two separate input contracts -- this is the
important part, they are NOT the same 43 columns fed in the same way:

  models/estrus_xgb_model.json
    - Objective: reg:squarederror (regression)
    - num_feature = 43, but the file has NO feature_names saved.
      XGBoost falls back to pure column POSITION when a booster has no
      stored names, so the input array must be in *exactly* the pipeline
      column order (RAW_43_COLUMNS below) every single call -- there is
      no name-based safety net for this model.
    - Predicts a single continuous value per row (days_to_estrus, per
      the training pipeline that produced this file).

  models/estrus_24hour_xgb_model_phase1.json
    - Objective: binary:logistic (classifier), heavily imbalanced
      (scale_pos_weight ~18) -- predicts P(estrus within 24h).
    - num_feature = 45. Its stored feature_names differ from the 43-col
      pipeline output in two ways, both handled here:
        1. It expects "body_temperature_C" at position 2 (right after
           THI), not "predicted_body_temp_C" at the end. We alias/move
           it -- there is no real body-temperature sensor on the
           wearable, predicted_body_temp_C (Model 1's output) is the
           only body-temp signal available at inference time.
        2. It expects two extra engineered flags not present in the
           43-col payload: thermal_inertia_lag_12h_was_missing and
           walking_hours_zscore_48h_was_missing. These are derived here
           as 1/0 indicators of whether those two columns were NaN.
      Because this booster DOES have feature_names saved, we read them
      from the loaded model at import time and reindex the input
      DataFrame to match exactly -- this is the safer of the two models
      to call since a name mismatch will raise instead of silently
      feeding misaligned columns.

Neither model receives imputed/filled values beyond the two explicit
"was_missing" flags above -- any remaining NaN (e.g. body_temp_delta_6h
before hour 6, from the structural warm-up period of a fresh 168h
window) is left as NaN and handled by XGBoost's native missing-value
routing (DMatrix's default missing=NaN), not filled with a guessed
constant.

CAVEAT (read before trusting predictions in production): the reference
training pipeline for estrus_xgb_model.json also saved
estrus_imputer.pkl and estrus_scaler.pkl. If those were applied to the
features before training, this module is NOT reproducing that
preprocessing -- we don't have those artifacts. If predictions look
off, that mismatch is the first thing to check.
"""

import os
import numpy as np
import xgboost as xgb
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DAYS_MODEL_PATH = os.path.join(MODELS_DIR, "estrus_xgb_model.json")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "estrus_24hour_xgb_model_phase1.json")

# Exact order the days-to-estrus model was trained on. This booster has
# no feature_names saved, so position is the ONLY contract -- do not
# reorder this list without re-checking against the model file.
RAW_43_COLUMNS = [
    "region_temperature_C", "THI", "heart_rate_bpm", "Standing", "Walking",
    "Eating", "Ruminating", "neck_temperature_C", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "predicted_body_temp_C",
    "body_temp_zscore_24h", "body_temp_zscore_6h", "body_temp_delta_6h",
    "neck_temp_zscore_24h", "thermal_gradient_body_neck",
    "thi_adjusted_body_temp_residual", "cardiothermal_index",
    "heart_rate_zscore_12h", "heart_rate_volatility_6h",
    "walking_hours_6h", "walking_hours_zscore_48h", "standing_hours_12h",
    "rumination_deficit_24h", "eating_deficit_12h", "restlessness_index_6h",
    "activity_to_rumination_ratio_12h", "postural_vigilance_ratio_12h",
    "walking_velocity_6h", "rumination_velocity_6h",
    "body_temp_acceleration_12h", "ema_crossover_walking_6_24",
    "cusum_activity_deviation_24h", "lag_24h_walking",
    "diurnality_index_24h", "nocturnal_walking_ratio_24h",
    "circadian_rumination_entropy_24h", "activity_thermal_spike_index",
    "metabolic_exertion_ratio", "cyclical_phase_interaction",
    "thermal_inertia_lag_12h",
]

_days_booster = None
_classifier_booster = None


def _load(path: str) -> xgb.Booster:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Place it in the models/ folder."
        )
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def get_days_model() -> xgb.Booster:
    global _days_booster
    if _days_booster is None:
        _days_booster = _load(DAYS_MODEL_PATH)
        if _days_booster.num_features() != len(RAW_43_COLUMNS):
            raise ValueError(
                f"estrus_xgb_model.json expects {_days_booster.num_features()} "
                f"features but RAW_43_COLUMNS has {len(RAW_43_COLUMNS)}. "
                "The model file and this module's column list are out of sync."
            )
    return _days_booster


def get_classifier_model() -> xgb.Booster:
    global _classifier_booster
    if _classifier_booster is None:
        _classifier_booster = _load(CLASSIFIER_MODEL_PATH)
        if not _classifier_booster.feature_names:
            raise ValueError(
                "estrus_24hour_xgb_model_phase1.json unexpectedly has no "
                "feature_names saved -- the name-based reindexing this "
                "module relies on can't run. Check the model file."
            )
    return _classifier_booster


def predict_days_to_estrus(features_df: pd.DataFrame) -> pd.Series:
    """
    features_df: DataFrame with (at least) all RAW_43_COLUMNS. Column
    order in features_df doesn't matter -- we select+reorder here -- but
    the underlying booster has no name metadata, so RAW_43_COLUMNS above
    must itself be correct and in the model's true training order.
    Returns a pandas Series of predicted days_to_estrus, one per row.
    """
    missing = [c for c in RAW_43_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"predict_days_to_estrus: missing required columns: {missing}")

    ordered = features_df[RAW_43_COLUMNS].astype(float)
    dmatrix = xgb.DMatrix(ordered, missing=np.nan)

    booster = get_days_model()
    preds = booster.predict(dmatrix)
    return pd.Series(preds, index=features_df.index, name="predicted_days_to_estrus")


def _build_classifier_input(features_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Builds the 45-column frame the classifier expects: renames
    predicted_body_temp_C -> body_temperature_C, derives the two
    was_missing flags, then reindexes to the model's own saved column
    order (name-based, safer than the days model's positional contract)."""
    df = features_df.copy()

    if "body_temperature_C" not in df.columns:
        if "predicted_body_temp_C" not in df.columns:
            raise ValueError(
                "_build_classifier_input: need either 'body_temperature_C' "
                "or 'predicted_body_temp_C' in the input."
            )
        df["body_temperature_C"] = df["predicted_body_temp_C"]

    if "thermal_inertia_lag_12h_was_missing" not in df.columns:
        df["thermal_inertia_lag_12h_was_missing"] = df["thermal_inertia_lag_12h"].isna().astype(int)
    if "walking_hours_zscore_48h_was_missing" not in df.columns:
        df["walking_hours_zscore_48h_was_missing"] = df["walking_hours_zscore_48h"].isna().astype(int)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"_build_classifier_input: missing required columns: {missing}")

    return df[feature_names]


def predict_estrus_probability_24h(features_df: pd.DataFrame) -> pd.Series:
    """
    features_df: DataFrame with (at least) all RAW_43_COLUMNS (using
    predicted_body_temp_C -- it gets aliased to body_temperature_C here).
    Returns a pandas Series of P(estrus within next 24h), one per row,
    in [0, 1] -- binary:logistic booster.predict() returns probabilities
    directly, not raw margins.
    """
    booster = get_classifier_model()
    ordered = _build_classifier_input(features_df, booster.feature_names).astype(float)
    dmatrix = xgb.DMatrix(ordered, feature_names=booster.feature_names, missing=np.nan)

    preds = booster.predict(dmatrix)
    return pd.Series(preds, index=features_df.index, name="estrus_probability_24h")
