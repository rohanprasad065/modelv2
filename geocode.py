"""
geocode.py
----------
Reverse-geocodes a location (lat/lon) to the Indian state it falls in,
using BigDataCloud's free reverse-geocoding endpoint (no API key
required, no signup, explicitly permits automated/server-side calls --
unlike Nominatim's public server, which returns 403 for exactly this
kind of programmatic use, regardless of what User-Agent you send).

https://www.bigdatacloud.com/geocoding-apis/free-reverse-geocode-to-city-api

Free tier: ~1500 requests/day (soft-blocked, not hard-failed, if
exceeded -- see the graceful-degradation handling below). At one lookup
per distinct location per week, this comfortably covers a large fleet.

Cached per rounded coordinate at module level -- a whole week's readings
from one animal are almost always the same location, so this normally
means ONE network call per request, not 168.
"""

import time
import requests

REVERSE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
HEADERS = {"User-Agent": "wearable-pipeline/1.0"}

_state_cache: dict[tuple, str] = {}


def _round_coord(x, precision=2):
    return round(x, 2)


def get_indian_state(lat: float, lon: float, retries: int = 2) -> str | None:
    """
    Returns the Indian state name for a lat/lon, or None if it can't be
    determined (API failure, or the point isn't in India).
    """
    key = (_round_coord(lat), _round_coord(lon))
    if key in _state_cache:
        return _state_cache[key]

    params = {"latitude": lat, "longitude": lon, "localityLanguage": "en"}
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(REVERSE_URL, params=params, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            country_code = (data.get("countryCode") or "").lower()
            state = data.get("principalSubdivision")
            result = state if country_code == "in" and state else None
            _state_cache[key] = result
            return result
        except Exception as e:  # noqa: BLE001 - degrade gracefully, don't crash the pipeline
            last_err = e
            time.sleep(0.5 * (attempt + 1))

    print(f"[geocode] failed to reverse-geocode ({lat},{lon}): {last_err}")
    _state_cache[key] = None
    return None
