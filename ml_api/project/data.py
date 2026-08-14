"""PostgREST data loading, shared by the standalone trainer and the backtest harness.

``CPHAPI_HOST`` is the full URL of the ``all`` endpoint (matching the previous
serving code), e.g. ``http://apisix:9080/api/v1/all`` internally or
``https://waitport.com/api/v1/all`` publicly.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from features import RESAMPLE_FREQUENCY, VALID_AIRPORTS

DEFAULT_BASE_URL = os.environ.get("CPHAPI_HOST", "https://waitport.com/api/v1/all")


def fetch_series(base_url: str, airport: str, start_utc=None, timeout: int = 120) -> pd.Series:
    """Fetch one airport's queue history as a 5-min-gridded Series.

    Returns a Series named ``queue`` indexed by a tz-aware UTC DatetimeIndex,
    de-duplicated by flooring to the resample grid and averaging collisions.
    Gaps are simply absent rows; callers decide how to handle them.
    """
    url = (
        f"{base_url}?select=queue,timestamp&airport=eq.{airport}"
        f"&order=timestamp.asc&limit=1000000"
    )
    if start_utc is not None:
        start_param = pd.Timestamp(start_utc).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        url += f"&timestamp=gte.{start_param}"

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.Series(dtype=float, name="queue")

    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    queue = pd.to_numeric(df["queue"], errors="coerce")
    df = pd.DataFrame({"queue": queue.to_numpy()}, index=ts).dropna()
    df = df[df["queue"] < 1e10]

    floored = df.index.floor(RESAMPLE_FREQUENCY)
    series = df.groupby(floored)["queue"].mean().sort_index()
    series.name = "queue"
    series.index.name = "timestamp"
    return series


def load_histories(base_url: str, airports=None, days=None) -> dict:
    """Fetch gridded histories for several airports -> {airport: Series}."""
    airports = airports or VALID_AIRPORTS
    start = None
    if days is not None:
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return {ap: fetch_series(base_url, ap, start) for ap in airports}
