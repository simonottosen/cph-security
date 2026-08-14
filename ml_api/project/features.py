"""Shared, side-effect-free feature engineering for the XGBoost calendar predictor.

Every feature depends only on the target timestamp's calendar attributes plus
seasonal *profiles* derived from history strictly before a cutoff. This makes
each feature valid for an arbitrary future timestamp (no row-based lags), which
removes the train/serve skew of the previous lag-based pipeline and lets a
single global model with a categorical ``airport`` column serve all airports.

The module has no import side effects so it can be reused by the serving app,
a standalone trainer, and the offline backtest harness alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import holidays

# Human travel patterns follow local wall-clock time; we key every calendar
# feature off Copenhagen local time for consistency with the incumbent model.
LOCAL_TZ = "Europe/Copenhagen"
RESAMPLE_FREQUENCY = "5min"
MINUTE_BUCKET = 5

# Airport -> holidays factory (called with an explicit year range so the
# resulting calendar is fully populated and membership tests are reliable).
_HOLIDAY_FACTORIES = {
    "AMS": holidays.Netherlands,
    "ARN": holidays.Sweden,
    "CPH": holidays.Denmark,
    "DUB": holidays.Ireland,
    "DUS": holidays.Germany,
    "FRA": holidays.Germany,
    "IST": holidays.Turkey,
    "LHR": holidays.UnitedKingdom,
    "EDI": holidays.UnitedKingdom,
    "MUC": holidays.Germany,
}
VALID_AIRPORTS = list(_HOLIDAY_FACTORIES.keys())

FEATURE_COLUMNS = [
    "month",
    "day",
    "weekday",
    "hour",
    "minute",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "tod_sin",
    "tod_cos",
    "holiday",
    "profile_dow_time",
    "profile_time",
]


@dataclass
class Profiles:
    """Seasonal climatology derived from an airport's history (leak-free)."""

    airport: str
    profile_dow_time: pd.Series  # mean queue indexed by (weekday, hour, minute)
    profile_time: pd.Series      # mean queue indexed by (hour, minute)
    global_mean: float


def _to_local(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(LOCAL_TZ)


def _holiday_set(airport: str, years) -> set:
    factory = _HOLIDAY_FACTORIES.get(airport)
    if factory is None:
        return set()
    return set(factory(years=list(years)).keys())


def build_profiles(history: pd.Series, airport: str) -> Profiles:
    """Build seasonal profiles from a queue history Series with a datetime index.

    ``history`` should contain only observations at/before the cutoff so the
    profiles never see the future.
    """
    if history is None or len(history) == 0:
        empty = pd.Series(dtype=float)
        return Profiles(airport, empty, empty, 0.0)

    local = _to_local(history.index)
    frame = pd.DataFrame(
        {
            "queue": np.asarray(history.to_numpy(), dtype=float),
            "weekday": local.weekday,
            "hour": local.hour,
            "minute": (local.minute // MINUTE_BUCKET) * MINUTE_BUCKET,
        }
    )
    profile_dow_time = frame.groupby(["weekday", "hour", "minute"])["queue"].mean()
    profile_time = frame.groupby(["hour", "minute"])["queue"].mean()
    global_mean = float(frame["queue"].mean())
    return Profiles(airport, profile_dow_time, profile_time, global_mean)


def compute_features(target_index, profiles: Profiles) -> pd.DataFrame:
    """Feature frame for a set of target timestamps belonging to one airport."""
    idx = pd.DatetimeIndex(target_index)
    local = _to_local(idx)

    weekday = local.weekday.to_numpy()
    hour = local.hour.to_numpy()
    raw_minute = local.minute.to_numpy()
    minute = (raw_minute // MINUTE_BUCKET) * MINUTE_BUCKET
    minute_of_day = hour * 60 + raw_minute

    out = pd.DataFrame(index=idx)
    out["month"] = local.month.to_numpy()
    out["day"] = local.day.to_numpy()
    out["weekday"] = weekday
    out["hour"] = hour
    out["minute"] = minute
    out["is_weekend"] = (weekday >= 5).astype(np.int8)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    if len(idx):
        years = range(int(local.year.min()), int(local.year.max()) + 1)
    else:
        years = range(2020, 2021)
    hol_set = _holiday_set(profiles.airport, years)
    dates = local.date
    out["holiday"] = np.fromiter(
        (1 if d in hol_set else 0 for d in dates), dtype=np.int8, count=len(dates)
    )

    # Profile lookups with graceful fallback: (dow, time) -> (time) -> global mean.
    n = len(idx)
    if len(profiles.profile_dow_time):
        keys_dow = pd.MultiIndex.from_arrays(
            [weekday, hour, minute], names=["weekday", "hour", "minute"]
        )
        p_dow = profiles.profile_dow_time.reindex(keys_dow).to_numpy(dtype=float)
    else:
        p_dow = np.full(n, np.nan)
    if len(profiles.profile_time):
        keys_time = pd.MultiIndex.from_arrays([hour, minute], names=["hour", "minute"])
        p_time = profiles.profile_time.reindex(keys_time).to_numpy(dtype=float)
    else:
        p_time = np.full(n, np.nan)

    p_dow = np.where(np.isnan(p_dow), p_time, p_dow)
    p_dow = np.where(np.isnan(p_dow), profiles.global_mean, p_dow)
    p_time = np.where(np.isnan(p_time), profiles.global_mean, p_time)
    out["profile_dow_time"] = p_dow
    out["profile_time"] = p_time

    return out[FEATURE_COLUMNS]


def build_training_frame(history_by_airport, profiles_by_airport):
    """Assemble a global training frame (features + ``airport`` categorical, y)."""
    x_parts, y_parts = [], []
    for airport, hist in history_by_airport.items():
        if hist is None or len(hist) == 0:
            continue
        feats = compute_features(hist.index, profiles_by_airport[airport]).copy()
        feats["airport"] = airport
        x_parts.append(feats)
        y_parts.append(pd.Series(np.asarray(hist.to_numpy(), dtype=float), index=hist.index))

    if not x_parts:
        empty_x = pd.DataFrame(columns=FEATURE_COLUMNS + ["airport"])
        return empty_x, pd.Series(dtype=float)

    x = pd.concat(x_parts)
    y = pd.concat(y_parts)
    x["airport"] = pd.Categorical(x["airport"], categories=VALID_AIRPORTS)
    return x, y
