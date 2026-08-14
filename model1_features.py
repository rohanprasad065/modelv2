"""
model1_features.py
-------------------
Converts a single clean 168-hour window (the 12-column output of our
main pipeline: region_temperature_C, THI, heart_rate_bpm, Standing,
Walking, Eating, Ruminating, neck_temperature_C, hour_sin, hour_cos,
day_sin, day_cos) into the 36-feature set Model 1 was trained on, per
`Model1_raw_to_features_conversion.md`.

Every formula uses only i <= t (fully causal) and is computed fresh for
this window -- exactly matching how a live 7-day inference window has no
memory of any other window. `body_temperature_C` and `estrus` are never
read or produced here: Model 1's whole purpose is predicting body temp
from sensors that don't include it.

Validated row-for-row against the provided training reference
(1786640125967_model1_body_temp_features__1_.csv, window "0_0") --
every one of the 36 derived columns matches to floating-point precision.
"""

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "region_temperature_C", "THI", "heart_rate_bpm",
    "Standing", "Walking", "Eating", "Ruminating",
    "neck_temperature_C", "hour_sin", "hour_cos", "day_sin", "day_cos",
]

FEATURE_COLUMNS = RAW_COLUMNS + [
    "lag_1h_neck", "lag_2h_neck", "lag_3h_neck", "lag_24h_neck",
    "diff_1h_neck", "diff_1h_hr",
    "lag_1h_hr", "lag_2h_hr",
    "rolling_mean_6h_neck", "rolling_std_6h_neck",
    "rolling_mean_24h_neck", "rolling_std_24h_neck",
    "rolling_mean_6h_hr", "rolling_std_6h_hr",
    "rolling_mean_24h_hr", "rolling_std_24h_hr",
    "expanding_mean_neck", "expanding_std_neck",
    "THI_rolling_mean_24h", "heat_stress_hours_24h",
    "active_hours_6h",
    "THI_activity", "temp_gradient",
    "hours_elapsed",
]


def build_model1_features(window: pd.DataFrame) -> pd.DataFrame:
    """
    window: DataFrame with exactly RAW_COLUMNS, in chronological order,
    representing one independent 168-hour block (row 0 = t=0 for all
    lag/rolling/expanding formulas -- no state carries in from outside).
    Returns a new DataFrame with all 36 FEATURE_COLUMNS, same row count.
    """
    missing = [c for c in RAW_COLUMNS if c not in window.columns]
    if missing:
        raise ValueError(f"build_model1_features: missing required raw columns: {missing}")

    df = window[RAW_COLUMNS].reset_index(drop=True).copy()
    neck = df["neck_temperature_C"]
    hr = df["heart_rate_bpm"]
    thi = df["THI"]
    active = df["Standing"].astype(float) + df["Walking"].astype(float)

    # --- Group 7: expanding stats (computed early -- used as the fill
    # value for the structural lag NaNs in Group 2) ---
    expanding_mean_neck = neck.expanding(min_periods=1).mean()
    expanding_std_neck = neck.expanding(min_periods=1).std(ddof=0)

    # --- Group 2: neck temp lags, structural NaNs filled with expanding_mean_neck(t) ---
    lag_1h_neck = neck.shift(1).fillna(expanding_mean_neck)
    lag_2h_neck = neck.shift(2).fillna(expanding_mean_neck)
    lag_3h_neck = neck.shift(3).fillna(expanding_mean_neck)
    lag_24h_neck = neck.shift(24).fillna(expanding_mean_neck)

    # --- Group 3: rate of change, = 0 at t=0 ---
    diff_1h_neck = neck.diff().fillna(0.0)
    diff_1h_hr = hr.diff().fillna(0.0)

    # --- Group 4: heart rate lags, structural NaNs filled with FIRST hr value ---
    first_hr = hr.iloc[0]
    lag_1h_hr = hr.shift(1).fillna(first_hr)
    lag_2h_hr = hr.shift(2).fillna(first_hr)

    # --- Group 5: rolling stats, neck temp (window caps at available history) ---
    rolling_mean_6h_neck = neck.rolling(window=6, min_periods=1).mean()
    rolling_std_6h_neck = neck.rolling(window=6, min_periods=1).std(ddof=0)
    rolling_mean_24h_neck = neck.rolling(window=24, min_periods=1).mean()
    rolling_std_24h_neck = neck.rolling(window=24, min_periods=1).std(ddof=0)

    # --- Group 6: rolling stats, heart rate ---
    rolling_mean_6h_hr = hr.rolling(window=6, min_periods=1).mean()
    rolling_std_6h_hr = hr.rolling(window=6, min_periods=1).std(ddof=0)
    rolling_mean_24h_hr = hr.rolling(window=24, min_periods=1).mean()
    rolling_std_24h_hr = hr.rolling(window=24, min_periods=1).std(ddof=0)

    # --- Group 8: heat-load accumulation ---
    THI_rolling_mean_24h = thi.rolling(window=24, min_periods=1).mean()
    heat_stress_hours_24h = (thi > 72).astype(float).rolling(window=24, min_periods=1).sum()

    # --- Group 9: activity accumulation ---
    active_hours_6h = active.rolling(window=6, min_periods=1).sum()

    # --- Group 10: interaction terms ---
    THI_activity = thi * active
    temp_gradient = df["region_temperature_C"] - neck

    # --- Group 11: position in window ---
    hours_elapsed = np.arange(len(df), dtype=float)

    out = df.copy()
    out["lag_1h_neck"] = lag_1h_neck
    out["lag_2h_neck"] = lag_2h_neck
    out["lag_3h_neck"] = lag_3h_neck
    out["lag_24h_neck"] = lag_24h_neck
    out["diff_1h_neck"] = diff_1h_neck
    out["diff_1h_hr"] = diff_1h_hr
    out["lag_1h_hr"] = lag_1h_hr
    out["lag_2h_hr"] = lag_2h_hr
    out["rolling_mean_6h_neck"] = rolling_mean_6h_neck
    out["rolling_std_6h_neck"] = rolling_std_6h_neck
    out["rolling_mean_24h_neck"] = rolling_mean_24h_neck
    out["rolling_std_24h_neck"] = rolling_std_24h_neck
    out["rolling_mean_6h_hr"] = rolling_mean_6h_hr
    out["rolling_std_6h_hr"] = rolling_std_6h_hr
    out["rolling_mean_24h_hr"] = rolling_mean_24h_hr
    out["rolling_std_24h_hr"] = rolling_std_24h_hr
    out["expanding_mean_neck"] = expanding_mean_neck
    out["expanding_std_neck"] = expanding_std_neck
    out["THI_rolling_mean_24h"] = THI_rolling_mean_24h
    out["heat_stress_hours_24h"] = heat_stress_hours_24h
    out["active_hours_6h"] = active_hours_6h
    out["THI_activity"] = THI_activity
    out["temp_gradient"] = temp_gradient
    out["hours_elapsed"] = hours_elapsed

    return out[FEATURE_COLUMNS]
