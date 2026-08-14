"""Unit tests for the shared calendar/profile feature builder.

These lock in the behaviours the calendar predictor depends on: a stable
feature schema, DST-correct local-time conversion, holiday flagging, and
leak-free seasonal profiles with graceful fallbacks.
"""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

# Make ml_api/project importable when run standalone (mirrors conftest.py).
_PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"
)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import features  # noqa: E402


def _sample_history(n=3000, start="2026-04-01", tz="UTC", base=6.0):
    idx = pd.date_range(start, periods=n, freq="5min", tz=tz)
    vals = base + 3.0 * np.sin(np.arange(n) / 40.0)
    return pd.Series(vals, index=idx, name="queue")


def test_feature_schema_is_stable():
    s = _sample_history()
    prof = features.build_profiles(s, "CPH")
    feats = features.compute_features(s.index[:10], prof)
    assert list(feats.columns) == features.FEATURE_COLUMNS
    assert len(feats) == 10
    assert not feats.isna().any().any()


def test_local_time_conversion_is_dst_aware():
    prof = features.build_profiles(_sample_history(), "CPH")
    # Winter: CET is UTC+1; summer: CEST is UTC+2.
    winter = features.compute_features([pd.Timestamp("2026-01-15T12:00", tz="UTC")], prof)
    summer = features.compute_features([pd.Timestamp("2026-07-15T12:00", tz="UTC")], prof)
    assert int(winter["hour"].iloc[0]) == 13
    assert int(summer["hour"].iloc[0]) == 14


def test_holiday_flag_fires_for_local_holiday():
    prof = features.build_profiles(_sample_history(), "CPH")
    # 1 Jan is a Danish public holiday; 17 Mar 2026 is an ordinary weekday.
    holiday = features.compute_features([pd.Timestamp("2026-01-01T10:00", tz="UTC")], prof)
    ordinary = features.compute_features([pd.Timestamp("2026-03-17T10:00", tz="UTC")], prof)
    assert int(holiday["holiday"].iloc[0]) == 1
    assert int(ordinary["holiday"].iloc[0]) == 0


def test_naive_timestamps_are_treated_as_utc():
    prof = features.build_profiles(_sample_history(), "CPH")
    aware = features.compute_features([pd.Timestamp("2026-07-15T12:00", tz="UTC")], prof)
    naive = features.compute_features([pd.Timestamp("2026-07-15T12:00")], prof)
    assert int(aware["hour"].iloc[0]) == int(naive["hour"].iloc[0]) == 14


def test_profile_falls_back_to_global_mean_for_unseen_slot():
    # History covering only a single weekday/time slot.
    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-01T00:00", tz="UTC")] )
    prof = features.build_profiles(pd.Series([9.0], index=idx, name="queue"), "CPH")
    assert prof.global_mean == 9.0
    # A completely different slot has no (dow,hour,minute) or (hour,minute) match.
    feats = features.compute_features([pd.Timestamp("2026-06-02T05:30", tz="UTC")], prof)
    assert feats["profile_dow_time"].iloc[0] == 9.0
    assert feats["profile_time"].iloc[0] == 9.0


def test_empty_history_yields_finite_features():
    prof = features.build_profiles(pd.Series(dtype=float), "CPH")
    assert prof.global_mean == 0.0
    feats = features.compute_features([pd.Timestamp("2026-07-15T12:00", tz="UTC")], prof)
    assert np.isfinite(feats.to_numpy(dtype=float)).all()
    assert feats["profile_dow_time"].iloc[0] == 0.0


def test_build_training_frame_uses_categorical_airport():
    s_cph = _sample_history(base=6.0)
    s_arn = _sample_history(base=9.0)
    profs = {
        "CPH": features.build_profiles(s_cph, "CPH"),
        "ARN": features.build_profiles(s_arn, "ARN"),
    }
    x, y = features.build_training_frame({"CPH": s_cph, "ARN": s_arn}, profs)
    assert "airport" in x.columns
    assert str(x["airport"].dtype) == "category"
    assert list(x["airport"].cat.categories) == features.VALID_AIRPORTS
    assert len(x) == len(y) == len(s_cph) + len(s_arn)
    assert list(x.columns) == features.FEATURE_COLUMNS + ["airport"]


if __name__ == "__main__":
    # Allow running without pytest installed: python ml_api/tests/test_features.py
    import sys

    sys.path.insert(0, features.__file__.rsplit("/", 1)[0])
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {exc!r}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
