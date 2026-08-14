"""Unit tests for the deterministic calendar covariates (``_add_time_covariates``).

These covariates are *known for future timestamps*, so they feed both the
historical context and the ``future_df`` of the batched Chronos-2 call. The
tests lock in three properties the model relies on: the covariate set is stable,
the cyclical encodings stay within the unit circle, and the builder is pure (it
must not mutate the caller's frame, which is reused across airports).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _app():
    repo_root = Path(__file__).resolve().parents[2]  # <repo>/autogluon
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("project.app")


def _raw_df(periods=288):
    """A 24h, 5-minute frame with the ``timestamp`` column the builder expects."""
    ts = pd.date_range("2024-01-01", periods=periods, freq="5min")
    return pd.DataFrame({"timestamp": ts, "queue": np.random.rand(periods)})


def test_time_covariates_add_columns_non_destructively():
    app = _app()
    raw = _raw_df()
    out = app._add_time_covariates(raw)

    assert len(out) == len(raw)
    for col in raw.columns:
        assert col in out.columns
    for expected in ("hour", "day_of_week", "is_weekend", "month",
                     "tod_sin", "tod_cos", "tow_sin", "tow_cos", "doy_sin", "doy_cos"):
        assert expected in out.columns


def test_cyclical_columns_within_unit_circle():
    app = _app()
    out = app._add_time_covariates(_raw_df())

    trig_cols = [c for c in out.columns if c.endswith(("sin", "cos"))]
    assert trig_cols, "expected sine/cosine covariates"
    for col in trig_cols:
        series = out[col].dropna()
        assert (series.abs() <= 1.0 + 1e-9).all(), f"{col} outside [-1, 1]"


def test_calendar_columns_have_sane_ranges():
    app = _app()
    out = app._add_time_covariates(_raw_df())

    assert out["hour"].between(0, 23).all()
    assert out["month"].between(1, 12).all()
    assert set(out["is_weekend"].unique()) <= {0.0, 1.0}
    assert out["day_of_week"].between(0, 6).all()


def test_builder_does_not_mutate_input():
    app = _app()
    raw = _raw_df()
    raw_copy = raw.copy(deep=True)

    _ = app._add_time_covariates(raw)

    pd.testing.assert_frame_equal(raw, raw_copy, check_dtype=False)
