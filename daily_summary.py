"""
daily_summary.py
------------------
Collapses the 168-row hourly week_data table into 7 rows, one per day.

Per-day aggregation rules (as specified):
  date                    -> the day's date (unchanged)
  time                    -> None. A daily row has no single representative
                              hour -- max_region_temp and max_heart_rate_bpm
                              can each come from a different hour of the
                              day, so there is no one "time" value that's
                              honestly correct here. Left as null rather
                              than picking an arbitrary hour and implying
                              it means something it doesn't.
  max_region_temperature_C -> max(region_temperature_C) over the day's 24 hours
  max_THI                  -> max(THI) over the day
  max_heart_rate_bpm       -> max(heart_rate_bpm) over the day
  standing_pct             -> % of the day's 24 hours spent Standing
  walking_pct               -> % of the day's 24 hours spent Walking
  eating_pct                -> % of the day's 24 hours spent Eating
  ruminating_pct             -> % of the day's 24 hours spent Ruminating
                              (Standing/Walking/Eating/Ruminating are a
                              one-hot flag per hour -- exactly one is 1.0
                              each hour -- so these four percentages always
                              sum to exactly 100.)
  max_neck_temperature_C   -> max(neck_temperature_C) over the day
  max_predicted_body_temp_C -> max(predicted_body_temp_C) over the day
  max_humidity              -> max(humidity) over the day

`state` is intentionally dropped, per instructions.
"""

import pandas as pd

DAILY_COLUMNS = [
    "date",
    "time",
    "max_region_temperature_C",
    "max_THI",
    "max_heart_rate_bpm",
    "standing_pct",
    "walking_pct",
    "eating_pct",
    "ruminating_pct",
    "max_neck_temperature_C",
    "max_predicted_body_temp_C",
    "max_humidity",
]


def build_daily_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    hourly_df: DataFrame with (at least) the columns date, region_temperature_C,
    THI, heart_rate_bpm, Standing, Walking, Eating, Ruminating,
    neck_temperature_C, predicted_body_temp_C, humidity -- one row per hour,
    grouped 24-at-a-time by the 'date' column (as produced by main.py's
    week_data table). Returns one row per distinct date, in the same order
    dates first appear (i.e. chronological, matching the input).
    """
    required = [
        "date", "region_temperature_C", "THI", "heart_rate_bpm",
        "Standing", "Walking", "Eating", "Ruminating",
        "neck_temperature_C", "predicted_body_temp_C", "humidity",
    ]
    missing = [c for c in required if c not in hourly_df.columns]
    if missing:
        raise ValueError(f"build_daily_summary: missing required columns: {missing}")

    df = hourly_df.copy()
    # Activity flags may come in as booleans or 0/1 -- normalize to float
    # so .sum() gives an hour-count regardless of the incoming dtype.
    for col in ["Standing", "Walking", "Eating", "Ruminating"]:
        df[col] = df[col].astype(float)

    rows = []
    # groupby(sort=False) preserves first-seen (chronological) date order,
    # rather than re-sorting dates alphabetically/numerically.
    for date_val, day in df.groupby("date", sort=False):
        hours_in_day = len(day)  # normally 24; guards against a short/partial day
        rows.append({
            "date": date_val,
            "time": None,
            "max_region_temperature_C": float(day["region_temperature_C"].max()),
            "max_THI": float(day["THI"].max()),
            "max_heart_rate_bpm": float(day["heart_rate_bpm"].max()),
            "standing_pct": round(day["Standing"].sum() / hours_in_day * 100, 2),
            "walking_pct": round(day["Walking"].sum() / hours_in_day * 100, 2),
            "eating_pct": round(day["Eating"].sum() / hours_in_day * 100, 2),
            "ruminating_pct": round(day["Ruminating"].sum() / hours_in_day * 100, 2),
            "max_neck_temperature_C": float(day["neck_temperature_C"].max()),
            "max_predicted_body_temp_C": float(day["predicted_body_temp_C"].max()),
            "max_humidity": float(day["humidity"].max()) if day["humidity"].notna().any() else None,
        })

    return pd.DataFrame(rows, columns=DAILY_COLUMNS)
