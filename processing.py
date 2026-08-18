"""
processing.py
--------------
Takes raw, irregularly-spaced wearable readings (nominally every 30 min,
but actually anywhere from ~25 to ~45 min apart, with possible missing
columns per row) and produces a clean 7-day x 1-hour dataset (168 rows)
with no null values.

Design principles (standard for IoT/wearable pipelines):
1. Never assume fixed intervals -> bucket by wall-clock hour, not by count.
2. Aggregate whatever raw points fall in each hour; missing columns in a
   row don't kill the row (mean/mode skip NaNs automatically).
3. If a whole hour has zero raw points, that hour is filled by
   interpolation (numeric) / forward-fill (categorical) after resampling,
   NOT dropped -> guarantees exactly 168 rows out.
4. If a column has only ONE non-null value in the entire 7-day window,
   every missing value in that column is filled with that single value
   (per your spec) rather than interpolated, since interpolation needs
   at least 2 points to be meaningful.
5. Every output row carries quality metadata: how many raw points fed it,
   and whether it was imputed. Don't throw this away -- production
   systems keep it so downstream consumers can trust (or distrust) a row.
"""

import re
import numpy as np
import pandas as pd

NUMERIC_COLS = ["neck_temp", "heart_rate", "region_temp", "lat", "lon"]
CATEGORICAL_COLS = ["activity"]

# Fixed activity categories for the target schema -- one-hot output must
# always have exactly these 4 columns, even if a given week never contains
# one of them (e.g. an animal that never "Ruminating"s that week still
# needs a Ruminating column of all 0s, not a missing column).
ACTIVITY_CATEGORIES = ["Standing", "Walking", "Eating", "Ruminating"]

RAW_COLUMN_MAP = {
    "Date": "date",
    "Time": "time",
    "Temp (°C)": "neck_temp",
    "Temp (Â°C)": "neck_temp",   # guard against mojibake if file isn't utf-8
    "Heart Rate": "heart_rate",
    "Ambient Temp": "region_temp",
    "Activity": "activity",
    "Location(0.000000, 0.000000)": "location",
    "Location": "location",
}

LOCATION_RE = re.compile(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?")


def _split_location(val):
    """Parse '(lat, lon)' style strings into (lat, lon) floats. Returns (nan, nan) if unparsable/missing."""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan, np.nan
    m = LOCATION_RE.search(str(val))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def load_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw column names, combine Date+Time into a single datetime,
    and split the Location string into lat/lon floats.
    """
    df = df_raw.rename(columns={c: RAW_COLUMN_MAP.get(c, c) for c in df_raw.columns}).copy()

    # Combine date + time -> datetime. Coerce errors to NaT so garbage rows
    # don't crash the pipeline; we drop truly unparsable rows (no timestamp
    # means we can't place it in any hourly bucket at all).
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce",
            dayfirst=False,
        )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError("Input must contain Date+Time columns or a datetime column.")

    before = len(df)
    df = df.dropna(subset=["datetime"])
    dropped = before - len(df)

    if "location" in df.columns:
        lat_lon = df["location"].apply(_split_location)
        df["lat"] = lat_lon.apply(lambda t: t[0])
        df["lon"] = lat_lon.apply(lambda t: t[1])

    for col in ["neck_temp", "heart_rate", "region_temp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["datetime"] + [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns]
    df = df[keep].sort_values("datetime").reset_index(drop=True)

    df.attrs["rows_dropped_no_timestamp"] = dropped
    return df


def _mode_or_nan(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


def resample_to_hourly(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Bucket irregular raw readings into a fixed 168-row (days*24) hourly
    grid, anchored to the first observed timestamp floored to the hour.
    """
    if df.empty:
        raise ValueError("No valid rows to resample (all timestamps unparsable?).")

    df = df.set_index("datetime")

    start = df.index.min().floor("h")
    full_index = pd.date_range(start=start, periods=days * 24, freq="1h")

    # raw point count per hour -- quality signal, computed before any fill
    counts = df.resample("1h").size()

    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    present_cat = [c for c in CATEGORICAL_COLS if c in df.columns]

    hourly_numeric = df[present_numeric].resample("1h").mean() if present_numeric else pd.DataFrame(index=counts.index)
    hourly_cat = (
        df[present_cat].resample("1h").agg(_mode_or_nan) if present_cat else pd.DataFrame(index=counts.index)
    )

    hourly = hourly_numeric.join(hourly_cat, how="outer")
    hourly = hourly.reindex(full_index)
    counts = counts.reindex(full_index).fillna(0).astype(int)

    hourly["num_raw_points"] = counts.values

    all_cols = present_numeric + present_cat
    was_null_before_fill = hourly[all_cols].isna()

    # Rule: if a column has exactly one non-null value across the whole
    # 7-day window, use that single value as a constant fill (interpolation
    # is meaningless with only 1 known point).
    for col in all_cols:
        non_null = hourly[col].dropna()
        if len(non_null) == 1:
            hourly[col] = hourly[col].fillna(non_null.iloc[0])

    # Numeric: time-aware linear interpolation for internal gaps, then
    # edge-fill (ffill/bfill) for any leading/trailing gaps interpolation
    # can't reach.
    # NOTE: region_temp is deliberately EXCLUDED here. Priority order for
    # region_temp is: (1) actual raw hourly reading, (2) weather-API
    # lookup via that hour's location (handled in weather.py, run right
    # after this function), (3) only if location is also missing does it
    # fall back to interpolation/nearest-value -- that fallback pass runs
    # in weather.py too, after the API lookup, so it only touches rows
    # the API genuinely couldn't help.
    interpolate_cols = [c for c in present_numeric if c != "region_temp"]
    if interpolate_cols:
        hourly[interpolate_cols] = (
            hourly[interpolate_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    # Categorical: no numeric interpolation possible -> forward/backward fill
    # ("carry last known activity/nothing-changed assumption" is standard).
    if present_cat:
        hourly[present_cat] = hourly[present_cat].ffill().bfill()

    hourly["is_imputed"] = was_null_before_fill.any(axis=1).values

    hourly = hourly.reset_index().rename(columns={"index": "datetime"})
    return hourly


def clean_pipeline(df_raw: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    df = load_raw(df_raw)
    return resample_to_hourly(df, days=days)
