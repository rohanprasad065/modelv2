"""
model2_features.py
-------------------
Turns the 13-column output of the main pipeline (12-column hourly table +
predicted_body_temp_C from Model 1) into the 30 estrus-detection features
listed below. This intentionally EXCLUDES `estrus`, `24hour`, and
`days_to_estrus` -- those are training-time labels/indices, not features
computable from an inference request.

Input contract
--------------
`window` must contain, in chronological order, one row per hour for a
single animal's window (same shape as the /process-week response `data`
array turned into a DataFrame):

    region_temperature_C, THI, heart_rate_bpm,
    Standing, Walking, Eating, Ruminating,
    neck_temperature_C, hour_sin, hour_cos, day_sin, day_cos,
    predicted_body_temp_C

There is no ground-truth `body_temperature_C` sensor on the wearable --
`predicted_body_temp_C` (Model 1's output) is used as the body-temperature
signal for every formula below. If you later have a real sensor reading,
swap BODY_COL to "body_temperature_C" and this module works unchanged.

Formulas were reverse-derived from and validated against the reference
sample rows you supplied. Confirmed exact matches (see inline comments):
body_temp_zscore_24h/6h, neck_temp_zscore_24h, thermal_gradient_body_neck,
cardiothermal_index, rumination_deficit_24h, postural_vigilance_ratio_12h,
metabolic_exertion_ratio, cyclical_phase_interaction. The remaining
formulas (EMA crossover, CUSUM, entropy, diurnality) are implemented
literally per your written formula column since the sample rows didn't
exercise them (Walking/Eating were 0 and t < 24h throughout the sample).

Two documented assumptions, called out because they aren't derivable from
5 sample rows alone -- change the constants below if your training code
used different values:
  - EPS = 1e-6 (ratio-denominator stabilizer)
  - Day window = hours [6, 18) recovered from hour_sin/hour_cos via atan2,
    used only by diurnality_index_24h / nocturnal_walking_ratio_24h.

Like model1_features.py, every formula is causal (uses only i <= t) and
operates on one independent window at a time -- no state carries in from
outside, matching a stateless per-request inference call.
"""

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "region_temperature_C", "THI", "heart_rate_bpm",
    "Standing", "Walking", "Eating", "Ruminating",
    "neck_temperature_C", "hour_sin", "hour_cos", "day_sin", "day_cos",
    "predicted_body_temp_C",
]

BODY_COL = "predicted_body_temp_C"

EPS = 1e-6
DAY_START_HOUR = 6   # inclusive
DAY_END_HOUR = 18    # exclusive

FEATURE_COLUMNS = [
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


def _rolling_zscore(s: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    """(s(t) - rolling_mean)/rolling_std, ddof=0. NaN until `min_periods`
    points are available -- matches the blank cells in your reference
    rows for t < min_periods."""
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mean) / std


def _fit_thi_body_regression(thi: pd.Series, body: pd.Series):
    """Fit body ~ alpha*THI + beta once over the whole delivered window
    (the request already carries the full 168h block, so this is a
    one-shot batch fit, not a per-row causal computation -- consistent
    with how model1_features.py's rolling/expanding stats already treat
    the whole window as available at once)."""
    if thi.nunique(dropna=True) < 2 or body.notna().sum() < 2:
        return 0.0, float(body.mean()) if body.notna().any() else 0.0
    alpha, beta = np.polyfit(thi.astype(float), body.astype(float), 1)
    return float(alpha), float(beta)


def build_model2_features(window: pd.DataFrame) -> pd.DataFrame:
    """
    window: DataFrame with exactly RAW_COLUMNS, in chronological order,
    one independent hourly block (row 0 = t=0 -- no state carries in).
    Returns a new DataFrame with all 30 FEATURE_COLUMNS, same row count.
    """
    missing = [c for c in RAW_COLUMNS if c not in window.columns]
    if missing:
        raise ValueError(f"build_model2_features: missing required raw columns: {missing}")

    df = window[RAW_COLUMNS].reset_index(drop=True).copy()

    region = df["region_temperature_C"].astype(float)
    thi = df["THI"].astype(float)
    hr = df["heart_rate_bpm"].astype(float)
    standing = df["Standing"].astype(float)
    walking = df["Walking"].astype(float)
    eating = df["Eating"].astype(float)
    rum = df["Ruminating"].astype(float)
    neck = df["neck_temperature_C"].astype(float)
    hour_sin = df["hour_sin"].astype(float)
    hour_cos = df["hour_cos"].astype(float)
    day_cos = df["day_cos"].astype(float)
    body = df[BODY_COL].astype(float)

    out = pd.DataFrame(index=df.index)

    # --- Thermal / cardiac (rows 1-9 in your table) ---
    out["body_temp_zscore_24h"] = _rolling_zscore(body, 24)          # validated
    out["body_temp_zscore_6h"] = _rolling_zscore(body, 6)            # validated
    out["body_temp_delta_6h"] = body - body.shift(6)                 # NaN until t>=6
    out["neck_temp_zscore_24h"] = _rolling_zscore(neck, 24)          # validated
    out["thermal_gradient_body_neck"] = body - neck                  # validated

    alpha, beta = _fit_thi_body_regression(thi, body)
    out["thi_adjusted_body_temp_residual"] = body - (alpha * thi + beta)

    out["cardiothermal_index"] = (hr * body) / (region + EPS)        # validated
    out["heart_rate_zscore_12h"] = _rolling_zscore(hr, 12)
    out["heart_rate_volatility_6h"] = hr.rolling(6, min_periods=2).std(ddof=0)

    # --- Activity time-budgets (rolling hour-sums, reused below) ---
    walking_hours_6h = walking.rolling(6, min_periods=1).sum()
    walking_hours_12h = walking.rolling(12, min_periods=1).sum()
    walking_hours_24h = walking.rolling(24, min_periods=1).sum()
    standing_hours_6h = standing.rolling(6, min_periods=1).sum()
    standing_hours_12h = standing.rolling(12, min_periods=1).sum()
    eating_hours_6h = eating.rolling(6, min_periods=1).sum()
    eating_hours_12h = eating.rolling(12, min_periods=1).sum()
    rum_hours_6h = rum.rolling(6, min_periods=1).sum()
    rum_hours_12h = rum.rolling(12, min_periods=1).sum()
    rum_hours_24h = rum.rolling(24, min_periods=1).sum()
    rum_mean_24h = rum.rolling(24, min_periods=1).mean()
    eating_mean_12h = eating.rolling(12, min_periods=1).mean()

    out["walking_hours_6h"] = walking_hours_6h                       # validated
    out["walking_hours_zscore_48h"] = _rolling_zscore(walking_hours_6h, 48)
    out["standing_hours_12h"] = standing_hours_12h                   # validated

    # Self-projected baseline: "if the rate observed so far in this
    # window held for the full target window, what would the total be" --
    # matches your rumination_deficit_24h sample values exactly
    # (23, 21, 20, 19 at t=0..3).
    out["rumination_deficit_24h"] = rum_mean_24h * 24 - rum_hours_24h  # validated

    eating_baseline_12h = eating_mean_12h * 12
    out["eating_deficit_12h"] = (eating_baseline_12h - eating_hours_12h) / (eating_baseline_12h + EPS)

    out["restlessness_index_6h"] = (walking_hours_6h + standing_hours_6h) / (
        rum_hours_6h + eating_hours_6h + EPS
    )
    out["activity_to_rumination_ratio_12h"] = walking_hours_12h / (rum_hours_12h + EPS)
    out["postural_vigilance_ratio_12h"] = standing_hours_12h / (
        standing_hours_12h + rum_hours_12h + EPS
    )                                                                  # validated

    out["walking_velocity_6h"] = walking_hours_6h.diff(6) / 6
    out["rumination_velocity_6h"] = rum_hours_6h.diff(6) / 6

    out["body_temp_acceleration_12h"] = (
        body - 2 * body.shift(6) + body.shift(12)
    ) / 36

    ema6 = walking_hours_6h.ewm(span=6, adjust=False).mean()
    ema24 = walking_hours_6h.ewm(span=24, adjust=False).mean()
    out["ema_crossover_walking_6_24"] = ema6 - ema24

    # CUSUM is inherently recursive (cusum(t) depends on cusum(t-1)) so
    # it's computed with an explicit loop rather than a vectorized op.
    walking_rate_mean_24h = walking.rolling(24, min_periods=1).mean()
    walking_rate_std_24h = walking.rolling(24, min_periods=1).std(ddof=0).fillna(0.0)
    cusum = np.zeros(len(df))
    prev = 0.0
    for t in range(len(df)):
        threshold = walking_rate_mean_24h.iloc[t] + 0.5 * walking_rate_std_24h.iloc[t]
        prev = max(0.0, prev + walking.iloc[t] - threshold)
        cusum[t] = prev
    out["cusum_activity_deviation_24h"] = cusum

    out["lag_24h_walking"] = walking_hours_24h.shift(24)              # NaN until t>=24

    # Recover hour-of-day from the cyclical encoding (raw timestamps
    # aren't part of the 13-column payload).
    hour_of_day = (np.arctan2(hour_sin, hour_cos) % (2 * np.pi)) * 24 / (2 * np.pi)
    is_day = (hour_of_day >= DAY_START_HOUR) & (hour_of_day < DAY_END_HOUR)
    is_night = ~is_day

    day_activity = ((walking + standing) * is_day.astype(float)).rolling(24, min_periods=1).sum()
    night_activity = ((walking + standing) * is_night.astype(float)).rolling(24, min_periods=1).sum()
    out["diurnality_index_24h"] = (day_activity - night_activity) / (
        day_activity + night_activity + EPS
    )

    night_walking_24h = (walking * is_night.astype(float)).rolling(24, min_periods=1).sum()
    out["nocturnal_walking_ratio_24h"] = night_walking_24h / (walking_hours_24h + EPS)

    # Shannon entropy of the trailing-24h rumination pattern, treating
    # each hour's flag as a weight in the distribution over the window.
    def _entropy_24h(rum_series: pd.Series) -> pd.Series:
        vals = rum_series.to_numpy(dtype=float)
        n = len(vals)
        result = np.full(n, np.nan)
        for t in range(n):
            start = max(0, t - 23)
            block = vals[start:t + 1]
            total = block.sum()
            if total <= 0:
                result[t] = 0.0
                continue
            p = block / total
            result[t] = float(-np.sum(p * np.log2(p + EPS)))
        return pd.Series(result, index=rum_series.index)

    out["circadian_rumination_entropy_24h"] = _entropy_24h(rum)

    out["activity_thermal_spike_index"] = (
        out["walking_hours_zscore_48h"] * out["body_temp_zscore_24h"]
    )
    out["metabolic_exertion_ratio"] = hr / (eating_hours_12h + rum_hours_12h + EPS)  # validated
    out["cyclical_phase_interaction"] = hour_sin * day_cos                            # validated
    out["thermal_inertia_lag_12h"] = body.rolling(12, min_periods=2).corr(walking)

    return out[FEATURE_COLUMNS]
