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

FIXED 2026-08-20 (two bugs):
  1. RAW_43_COLUMNS had predicted_body_temp_C in the wrong position
     (end of the raw block instead of position 2, right after THI).
     Since this booster has no feature_names saved, it trusts pure
     column position -- the old order silently shifted every raw
     column from position 2 through 12 by one slot.
  2. The reference training pipeline for estrus_xgb_model.json fit a
     SimpleImputer + StandardScaler before training, but those objects
     were never saved -- only model.save_model(...) was called. This
     module previously fed raw, unscaled values straight into the
     model. estrus_imputer.pkl / estrus_scaler.pkl / estrus_feature_cols.pkl
     were reconstructed afterward (see
     generate_regressor_preprocessing_artifacts.py) by deterministically
     re-running the same fitting steps against the same training CSV --
     run that script once before starting this service if those files
     don't already exist in models/.

REMAINING KNOWN LIMITATION (not fixed by the above, requires retraining
to fully resolve): the 30 model2 engineered features are rolling/lag/
z-score computations. At training time they were computed as one
continuous series across each animal's full multi-thousand-hour
history; at inference time (this module, fed from main.py) they're
computed from a single isolated, memoryless 168-hour window. That's a
real difference in what the feature values represent, not just a code
bug -- it isn't something a preprocessing fix can correct, and it's the
likely cause of predictions being less reliable than the model's own
training-time metrics would suggest, particularly at longer horizons.
"""

import os
import joblib
import numpy as np
import xgboost as xgb
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DAYS_MODEL_PATH = os.path.join(MODELS_DIR, "estrus_xgb_model.json")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "estrus_24hour_xgb_model_phase1.json")
DAYS_IMPUTER_PATH = os.path.join(MODELS_DIR, "estrus_imputer.pkl")
DAYS_SCALER_PATH = os.path.join(MODELS_DIR, "estrus_scaler.pkl")
DAYS_FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "estrus_feature_cols.pkl")

# FIXED 2026-08-20: this list previously had predicted_body_temp_C at the
# end (position 12) instead of position 2. This booster has no
# feature_names saved, so it trusts pure column position -- the old order
# silently fed heart_rate_bpm into the slot the model was trained to read
# as body temperature, and shifted every raw column from position 2
# through 12 by one slot. Verified against the actual training CSV's
# natural column order (see generate_regressor_preprocessing_artifacts.py,
# which derives this same order programmatically from that file --
# feature_cols = [c for c in df.columns if c != 'days_to_estrus']).
# Do not reorder this list without re-checking against that script's
# printed order.
RAW_43_COLUMNS = [
    "region_temperature_C", "THI", "predicted_body_temp_C", "heart_rate_bpm",
    "Standing", "Walking", "Eating", "Ruminating", "neck_temperature_C",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
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
_days_imputer = None
_days_scaler = None
_days_feature_cols = None


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


def _get_days_preprocessing():
    """
    Loads the imputer/scaler/feature-order the days-to-estrus regressor
    was trained on. These were never saved during the original Colab
    training run -- only the model itself was. They were reconstructed
    afterward (see generate_regressor_preprocessing_artifacts.py) by
    deterministically re-running the same imputer/scaler-fitting steps
    against the same training CSV and the same train-slice split logic
    the original script used, which produces mathematically identical
    fitted values (spot-checked by hand against sklearn's own output).
    """
    global _days_imputer, _days_scaler, _days_feature_cols
    if _days_imputer is None or _days_scaler is None or _days_feature_cols is None:
        for path, label in [
            (DAYS_IMPUTER_PATH, "estrus_imputer.pkl"),
            (DAYS_SCALER_PATH, "estrus_scaler.pkl"),
            (DAYS_FEATURE_COLS_PATH, "estrus_feature_cols.pkl"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found at {path}. Run "
                    "generate_regressor_preprocessing_artifacts.py once to "
                    "create it before starting the service."
                )
        _days_imputer = joblib.load(DAYS_IMPUTER_PATH)
        _days_scaler = joblib.load(DAYS_SCALER_PATH)
        _days_feature_cols = joblib.load(DAYS_FEATURE_COLS_PATH)
    return _days_imputer, _days_scaler, _days_feature_cols


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

    FIXED 2026-08-20: previously this sent raw, unscaled physical-unit
    values straight into the DMatrix. The model was trained on
    median-imputed + StandardScaler-scaled features (its split thresholds
    live in standardized space), so this step was silently missing.
    Now applies the reconstructed imputer -> scaler (see
    _get_days_preprocessing) before prediction, same as the original
    training pipeline did.

    Returns a pandas Series of predicted days_to_estrus, one per row.
    """
    missing = [c for c in RAW_43_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"predict_days_to_estrus: missing required columns: {missing}")

    imputer, scaler, feature_cols = _get_days_preprocessing()

    # feature_cols (loaded from estrus_feature_cols.pkl) is the exact
    # order the imputer/scaler were fit on. It must match RAW_43_COLUMNS
    # position-for-position, with ONE deliberate exception: position 2 is
    # "body_temperature_C" in feature_cols (the real sensor column name
    # used at training time) but "predicted_body_temp_C" in
    # RAW_43_COLUMNS (there's no live sensor at inference, so Model 1's
    # prediction fills that slot -- same substitution already used for
    # the classifier). Any OTHER mismatch means the two have drifted out
    # of sync and predictions would silently be wrong again.
    for i, (a, b) in enumerate(zip(RAW_43_COLUMNS, feature_cols)):
        if a == b:
            continue
        if i == 2 and a == "predicted_body_temp_C" and b == "body_temperature_C":
            continue  # expected, deliberate substitution
        raise ValueError(
            f"predict_days_to_estrus: RAW_43_COLUMNS position {i} "
            f"('{a}') doesn't match estrus_feature_cols.pkl ('{b}'). "
            "Re-run generate_regressor_preprocessing_artifacts.py or fix "
            "RAW_43_COLUMNS -- they must stay in sync."
        )

    ordered = features_df[RAW_43_COLUMNS].astype(float)
    ordered = ordered.replace([np.inf, -np.inf], np.nan)

    # .values (not the DataFrame) is passed to imputer/scaler on purpose:
    # they were fit on column names from estrus_feature_cols.pkl, which
    # has "body_temperature_C" at position 2 -- but `ordered` here
    # legitimately has "predicted_body_temp_C" in that same slot (the
    # deliberate substitution checked above). Passing a DataFrame with
    # different column names would make sklearn raise a feature-name
    # mismatch error even though the *position* (what actually matters
    # for both the scaler and the position-only booster) is correct.
    imputed = imputer.transform(ordered.values)
    scaled = scaler.transform(imputed)

    dmatrix = xgb.DMatrix(scaled, missing=np.nan)

    booster = get_days_model()
    preds = booster.predict(dmatrix)
    preds = np.clip(preds, 0, None)  # days can't be negative
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
