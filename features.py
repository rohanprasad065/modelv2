"""
features.py
------------
Turns the clean, humidity-enriched hourly dataset into a model-ready
feature table:

1. heat_index_c: combines ambient_temp + humidity into a single
   "feels like" temperature using the NOAA/Rothfusz regression
   (industry-standard heat index formula).
2. One-hot encoding of the `activity` column.
3. Cyclical (sin/cos) encoding of time -- hour-of-day and day-of-week --
   so a model sees 23:00 and 00:00 as adjacent instead of far apart.
"""

import numpy as np
import pandas as pd


def compute_heat_index(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """
    NOAA heat index (Rothfusz regression). Formula is defined in °F, so we
    convert in and back out. Only meaningful above ~27°C / 80°F and with
    humidity >= 40%; below that we just return the ambient temp unchanged,
    which is what NOAA's own guidance recommends (the regression is not
    valid outside that range and produces nonsense numbers there).
    """
    temp_f = temp_c * 9 / 5 + 32
    rh = humidity_pct

    hi_simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
    avg_f = (temp_f + hi_simple) / 2

    hi_full = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f ** 2
        - 0.05481717 * rh ** 2
        + 0.00122874 * temp_f ** 2 * rh
        + 0.00085282 * temp_f * rh ** 2
        - 0.00000199 * temp_f ** 2 * rh ** 2
    )

    use_full = avg_f >= 80
    heat_index_f = np.where(use_full, hi_full, temp_f)
    heat_index_c = (heat_index_f - 32) * 5 / 9
    return pd.Series(heat_index_c, index=temp_c.index)


def add_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ambient_temp" in df.columns and "humidity" in df.columns:
        df["heat_index_c"] = compute_heat_index(df["ambient_temp"], df["humidity"])
    return df


def one_hot_activity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "activity" in df.columns:
        dummies = pd.get_dummies(df["activity"], prefix="activity")
        df = pd.concat([df, dummies], axis=1)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])

    hour = dt.dt.hour + dt.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = dt.dt.dayofweek  # Monday=0
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_heat_index(df)
    df = one_hot_activity(df)
    df = add_cyclical_time_features(df)
    return df
