"""
features.py
------------
Turns the clean, humidity-enriched hourly dataset into the target model
feature table:

  region_temperature_C, THI, heart_rate_bpm, Standing, Walking, Eating,
  Ruminating, neck_temperature_C, hour_sin, hour_cos, day_sin, day_cos

1. THI: livestock Temperature-Humidity Index (NRC 1971 formula), computed
   from region_temp + humidity. This replaces a generic "feels like"
   heat-index -- THI is the standard metric used in dairy/livestock heat
   stress monitoring, which is what this schema is for.
2. One-hot encoding of `activity` into 4 FIXED columns (Standing, Walking,
   Eating, Ruminating) -- fixed, not data-driven, so the output schema
   never changes shape even if one category is absent from a given week.
3. Cyclical (sin/cos) encoding of hour-of-day (24-cycle) AND day-of-year
   (365/366-cycle, NOT day-of-week). Verified against the target sample
   file: day_sin/day_cos there resolve to 365 distinct values, not 7.
"""

from processing import ACTIVITY_CATEGORIES
from geocode import get_indian_state
import numpy as np
import pandas as pd

FINAL_COLUMNS = [
    "region_temperature_C",
    "THI",
    "heart_rate_bpm",
    "Standing",
    "Walking",
    "Eating",
    "Ruminating",
    "neck_temperature_C",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    # -- appended, existing columns above are untouched --
    "humidity",
    "state",
]


def compute_thi(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """
    NRC (1971) livestock Temperature-Humidity Index:
        THI = (1.8*T + 32) - (0.55 - 0.0055*RH) * (1.8*T - 26)
    where T is dry-bulb air temp in °C and RH is relative humidity in %.
    This is the standard THI used in dairy heat-stress monitoring
    (values >= 68 indicate onset of heat stress, >= 80 severe stress).
    """
    t_f = 1.8 * temp_c + 32
    return t_f - (0.55 - 0.0055 * humidity_pct) * (t_f - 26)


def add_thi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "region_temp" in df.columns and "humidity" in df.columns:
        df["THI"] = compute_thi(df["region_temp"], df["humidity"])
    return df


def one_hot_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode `activity` into exactly ACTIVITY_CATEGORIES columns.
    Uses pd.Categorical with a fixed category list so the output always
    has all 4 columns (0-filled if a category never occurs), instead of
    pd.get_dummies' default of only emitting columns that appear in data.
    """
    df = df.copy()
    if "activity" in df.columns:
        cat = pd.Categorical(df["activity"], categories=ACTIVITY_CATEGORIES)
        dummies = pd.get_dummies(cat)
        dummies = dummies.reindex(columns=ACTIVITY_CATEGORIES, fill_value=False)
        dummies.index = df.index
        df = pd.concat([df, dummies], axis=1)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])

    hour = dt.dt.hour + dt.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day-OF-YEAR cycle (not day-of-week) -- matches target schema.
    # Use day + fractional hour for a smooth cycle across a 365.25-day year.
    day_of_year = dt.dt.dayofyear + hour / 24.0
    days_in_year = np.where(dt.dt.is_leap_year, 366, 365)
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    return df


def add_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse-geocodes each row's location to an Indian state name. Looks
    up once per distinct (rounded) location -- typically one lookup for
    the whole week, since an animal's location barely moves.

    Rows with no location for that specific hour (but where OTHER hours
    in the week do have one) inherit the nearest known state via
    ffill/bfill -- same fallback treatment region_temp gets in
    weather.py, since a state doesn't realistically change hour to hour.
    A row only ends up with state=None if EVERY hour in the week lacked
    location data, or if the resolved point genuinely isn't in India.
    """
    df = df.copy()
    if not {"lat", "lon"}.issubset(df.columns):
        df["state"] = None
        return df

    state = pd.Series([None] * len(df), index=df.index, dtype=object)
    has_loc = df["lat"].notna() & df["lon"].notna()
    for (lat_r, lon_r), idx in df[has_loc].groupby(
        [df.loc[has_loc, "lat"].round(2), df.loc[has_loc, "lon"].round(2)]
    ).groups.items():
        row = df.loc[idx[0]]
        state.loc[idx] = get_indian_state(row["lat"], row["lon"])

    # Fall back to the nearest hour's resolved state for hours that had
    # no location of their own that hour.
    if state.notna().any():
        state = state.ffill().bfill()

    df["state"] = state
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_thi(df)
    df = add_state(df)
    df = one_hot_activity(df)
    df = add_cyclical_time_features(df)

    rename_map = {"region_temp": "region_temperature_C", "neck_temp": "neck_temperature_C", "heart_rate": "heart_rate_bpm"}
    df = df.rename(columns=rename_map)

    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build final schema, missing columns: {missing}")

    return df[FINAL_COLUMNS]
