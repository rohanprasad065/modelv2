"""
weather.py
----------
Enriches the clean hourly dataset with humidity, pulled from a weather API
using each row's location + timestamp.

Uses Open-Meteo's free historical/archive endpoint (no API key required):
https://archive-api.open-meteo.com/v1/archive

Production notes:
- We batch by (rounded lat/lon, date) instead of calling the API once per
  row. A naive "one HTTP call per row" implementation would fire 168
  requests per device per week -- fine for a demo, a real rate-limit/cost
  problem at fleet scale. Grouping by day+location cuts that down to
  roughly (# distinct days) x (# distinct locations) calls.
- Coordinates are rounded to ~1km precision before grouping, since GPS
  jitter of a few meters shouldn't trigger a fresh API call.
- Any failed call degrades gracefully (humidity=NaN for that block) rather
  than crashing the whole pipeline -- one bad network call shouldn't lose
  a week of health data.
"""

import time
import requests
import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _round_coord(x, precision=2):
    # ~2 decimal places ≈ 1.1km. Good enough to group nearby readings
    # without spamming the API for GPS jitter.
    return round(x, precision)


def fetch_humidity_for_group(lat: float, lon: float, date: str, retries: int = 2) -> pd.DataFrame:
    """
    Fetch hourly relative humidity for one lat/lon on one calendar date.
    Returns a DataFrame with columns [datetime, humidity].
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "relative_humidity_2m",
        "timezone": "UTC",
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            humidity = hourly.get("relative_humidity_2m", [])
            return pd.DataFrame({
                "datetime": pd.to_datetime(times),
                "humidity": humidity,
            })
        except Exception as e:  # noqa: BLE001 - want graceful degradation, not a crash
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    print(f"[weather] failed to fetch humidity for ({lat},{lon}) on {date}: {last_err}")
    return pd.DataFrame({"datetime": [], "humidity": []})


def add_humidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'humidity' column to the hourly dataframe by grouping rows into
    (rounded lat, rounded lon, date) buckets and making one API call per
    bucket instead of per row.
    """
    df = df.copy()
    if not {"lat", "lon"}.issubset(df.columns):
        df["humidity"] = np.nan
        return df

    df["_lat_r"] = df["lat"].apply(lambda v: _round_coord(v) if pd.notna(v) else np.nan)
    df["_lon_r"] = df["lon"].apply(lambda v: _round_coord(v) if pd.notna(v) else np.nan)
    df["_date"] = df["datetime"].dt.strftime("%Y-%m-%d")

    df["humidity"] = np.nan
    groups = df.dropna(subset=["_lat_r", "_lon_r"]).groupby(["_lat_r", "_lon_r", "_date"])

    for (lat_r, lon_r, date), group_idx in groups.groups.items():
        rows = df.loc[group_idx]
        humidity_df = fetch_humidity_for_group(lat_r, lon_r, date)
        if humidity_df.empty:
            continue
        merged = pd.merge_asof(
            rows[["datetime"]].sort_values("datetime"),
            humidity_df.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )
        df.loc[rows.index, "humidity"] = merged["humidity"].values

    df = df.drop(columns=["_lat_r", "_lon_r", "_date"])

    # If some hours still missing humidity (API gap / no location that hour),
    # fall back to interpolation, same policy as the main cleaning stage.
    if df["humidity"].notna().sum() >= 1:
        df["humidity"] = df["humidity"].interpolate(limit_direction="both").ffill().bfill()

    return df
