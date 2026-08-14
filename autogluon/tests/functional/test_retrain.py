"""Functional tests for the batched, best-effort ``retrain()`` orchestration.

The forecaster no longer trains one model per airport in a process pool. A
single ``_run_forecast`` does one batched, cross-learning ``predict_df`` across
all airports, and ``retrain()`` is *best-effort*: on any failure it keeps the
last-good in-memory state instead of raising (so ``/forecast`` is never emptied
by a transient outage). These tests lock in that contract by stubbing the two
seams ``retrain()`` orchestrates — ``_load_raw`` and ``_run_forecast`` — so they
stay fast and independent of PostgREST or Chronos.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


@pytest.fixture
def app(monkeypatch, tmp_path):
    mod = importlib.import_module("project.app")
    # Persist into a throwaway dir so a successful retrain never touches the repo.
    monkeypatch.setattr(mod, "FORECAST_DIR", str(tmp_path), raising=True)
    return mod


def test_retrain_publishes_batched_results(app, monkeypatch):
    """A healthy run rebinds the global state and persists it to disk."""
    forecasts = {
        "CPH": [{"timestamp": "2025-01-01T00:00:00", "mean": 1.0, "q30": 0.8, "q70": 1.2}],
        "ARN": [{"timestamp": "2025-01-01T00:00:00", "mean": 2.0, "q30": 1.8, "q70": 2.2}],
    }
    metrics = {"CPH": {"cross_learning": True}, "ARN": {"cross_learning": True}}

    monkeypatch.setattr(app, "_load_raw", lambda: pd.DataFrame({"airport": ["CPH"]}), raising=True)
    monkeypatch.setattr(app, "_run_forecast", lambda df: (forecasts, metrics), raising=True)

    app.retrain()

    assert app.df_preds == forecasts
    assert app.train_metrics == metrics
    # Best-good state was persisted so a restart can serve it immediately.
    reloaded = app._load_persisted_forecasts()
    assert reloaded is not None
    assert reloaded["predictions"]["CPH"][0]["mean"] == 1.0


def test_retrain_keeps_last_good_on_load_failure(app, monkeypatch):
    """A PostgREST outage must not raise or wipe the previously served forecasts."""
    with app._state_lock:
        app.df_preds = {"CPH": [{"timestamp": "2025-01-01T00:00:00", "mean": 9.0}]}
        app.train_metrics = {"CPH": {"cross_learning": True}}
    last_good_preds = app.df_preds
    last_good_metrics = app.train_metrics

    def _boom():
        raise app.requests.ConnectionError("boom")

    monkeypatch.setattr(app, "_load_raw", _boom, raising=True)

    app.retrain()  # must not raise

    assert app.df_preds is last_good_preds
    assert app.train_metrics is last_good_metrics


def test_retrain_keeps_last_good_on_forecast_failure(app, monkeypatch):
    """If the batched predict blows up, keep serving the last-good forecasts."""
    with app._state_lock:
        app.df_preds = {"CPH": [{"timestamp": "2025-01-01T00:00:00", "mean": 9.0}]}
        app.train_metrics = {"CPH": {"cross_learning": True}}
    last_good_preds = app.df_preds

    def _explode(_df):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(app, "_load_raw", lambda: pd.DataFrame({"airport": ["CPH"]}), raising=True)
    monkeypatch.setattr(app, "_run_forecast", _explode, raising=True)

    app.retrain()  # must not raise

    assert app.df_preds is last_good_preds
