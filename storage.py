"""
storage.py
----------
Persists the intermediate 12-column clean hourly result (output of
processing + weather + THI, BEFORE Model 1's 36-feature construction)
to disk, so it can be reused later without re-running cleaning/weather
lookups -- e.g. to feed a different downstream model, re-derive features
with a different formula version, or audit what was actually sent.

Storage is a flat CSV per saved batch, named by a generated request_id.
Swap this for a real database (Postgres, S3, etc.) in production --
the save/load function signatures are the seam to do that behind.
"""

import os
import uuid
import datetime as dt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "hourly_base")


def save_hourly_base(df: pd.DataFrame, request_id: str | None = None) -> str:
    """
    Saves the 12-column hourly DataFrame to disk. Returns the request_id
    it was saved under (generated if not supplied).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    request_id = request_id or str(uuid.uuid4())
    filepath = os.path.join(DATA_DIR, f"{request_id}.csv")
    df.to_csv(filepath, index=False)

    # lightweight manifest line so saved batches are discoverable later
    manifest_path = os.path.join(DATA_DIR, "_manifest.csv")
    manifest_row = pd.DataFrame([{
        "request_id": request_id,
        "saved_at_utc": dt.datetime.utcnow().isoformat(),
        "rows": len(df),
        "columns": ",".join(df.columns),
    }])
    manifest_row.to_csv(
        manifest_path, mode="a", index=False, header=not os.path.exists(manifest_path)
    )
    return request_id


def load_hourly_base(request_id: str) -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, f"{request_id}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved hourly-base data found for request_id={request_id}")
    return pd.read_csv(filepath)


def list_saved() -> pd.DataFrame:
    manifest_path = os.path.join(DATA_DIR, "_manifest.csv")
    if not os.path.exists(manifest_path):
        return pd.DataFrame(columns=["request_id", "saved_at_utc", "rows", "columns"])
    df = pd.read_csv(manifest_path)
    # keep only the latest save per request_id (a batch may be re-saved
    # once with 12 columns, then again with predicted_body_temp_C added)
    return df.sort_values("saved_at_utc").drop_duplicates("request_id", keep="last")
