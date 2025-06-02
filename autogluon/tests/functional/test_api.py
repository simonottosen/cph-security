
"""
Functional tests for the Flask REST API defined in **project.app**.

Endpoints under test
--------------------
* ``GET /forecast/<airport>`` – returns JSON with a ``predictions`` list.
* ``GET /metrics``           – returns JSON object with the latest training metrics.

The tests stub out heavy dependencies (AutoGluon) *before* importing
`project.app` so that the import stays lightweight.
"""

from __future__ import annotations

import sys
import types
from importlib import import_module
from typing import Tuple

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
#                      Lightweight AutoGluon patch helper                     #
# --------------------------------------------------------------------------- #
def _ensure_dummy_autogluon(monkeypatch):
    """
    Ensure a minimal stub for ``autogluon.timeseries`` so that importing
    *project.app* does **not** pull in the heavyweight real package.
    """
    if "autogluon.timeseries" in sys.modules:
        return  # Stub already present (perhaps from another test run)

    ag_root = types.ModuleType("autogluon")
    ts_mod = types.ModuleType("autogluon.timeseries")

    class _DummyTSDF(dict):
        @classmethod
        def from_data_frame(cls, df, *_, **__):
            return cls({"dummy": df})

        def train_test_split(self, *_, **__):
            return self, self

    class _DummyPredictor:
        def __init__(self, *_, **__):
            pass

        def fit(self, *_, **__):
            return self

        def predict(self, *_, **__):
            idx = pd.date_range("2024-01-01", periods=1, freq="H")
            return pd.DataFrame(
                {"mean": [1.0], "0.3": [0.8], "0.7": [1.2]}, index=idx
            )

        def fit_summary(self, *_, **__):
            return {}

        def persist(self, *_, **__):
            pass

    ts_mod.TimeSeriesDataFrame = _DummyTSDF
    ts_mod.TimeSeriesPredictor = _DummyPredictor
    ag_root.timeseries = ts_mod

    monkeypatch.setitem(sys.modules, "autogluon", ag_root)
    monkeypatch.setitem(sys.modules, "autogluon.timeseries", ts_mod)


# --------------------------------------------------------------------------- #
#                               Pytest fixture                                #
# --------------------------------------------------------------------------- #
@pytest.fixture
def client_app(monkeypatch) -> Tuple:
    """
    Yield (Flask test client, imported project.app module) with the heavy
    dependencies stubbed out.
    """
    _ensure_dummy_autogluon(monkeypatch)
    app_module = import_module("project.app")

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        # Make sure shared mutable state is clean for every test.
        app_module.df_preds.clear()
        app_module.train_metrics.clear()
        yield client, app_module


# --------------------------------------------------------------------------- #
#                                   Tests                                     #
# --------------------------------------------------------------------------- #
def test_forecast_invalid_airport_returns_404(client_app):
    client, _ = client_app
    resp = client.get("/forecast/XYZ")
    assert resp.status_code == 404
    assert resp.is_json
    assert "error" in resp.get_json()


def test_forecast_valid_airport_no_data_yet_returns_empty_list(client_app):
    client, app_module = client_app
    airport = app_module.VALID_AIRPORTS[0]

    resp = client.get(f"/forecast/{airport}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["predictions"] == []


def test_forecast_valid_airport_with_data(client_app):
    client, app_module = client_app
    airport = app_module.VALID_AIRPORTS[0]

    # Inject synthetic prediction
    app_module.df_preds[airport] = [
        {"timestamp": "2025-01-01T00:00:00", "mean": 1, "q30": 0.8, "q70": 1.2}
    ]

    resp = client.get(f"/forecast/{airport}")
    assert resp.status_code == 200
    assert resp.get_json()["predictions"] == app_module.df_preds[airport]


def test_metrics_endpoint_returns_metrics(client_app):
    client, app_module = client_app

    # Insert fake metrics
    app_module.train_metrics["CPH"] = {"total_time_seconds": 1.23}

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["CPH"]["total_time_seconds"] == 1.23
