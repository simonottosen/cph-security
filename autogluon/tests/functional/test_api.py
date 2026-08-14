"""Functional tests for the Flask REST API defined in **project.app**.

Endpoints under test:

* ``GET /forecast/<airport>`` – JSON ``{"predictions": [...]}`` (404 on bad code).
* ``GET /metrics``           – JSON of the latest per-airport forecast metrics.

The module imports cleanly without Chronos/torch (it falls back to the naive
pipeline), so no heavyweight stubbing is required.
"""

from __future__ import annotations

from importlib import import_module
from typing import Tuple

import pytest


@pytest.fixture
def client_app() -> Tuple:
    """Yield (Flask test client, project.app module) with clean shared state."""
    app_module = import_module("project.app")
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        app_module.df_preds.clear()
        app_module.train_metrics.clear()
        yield client, app_module


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
    assert resp.get_json()["predictions"] == []


def test_forecast_valid_airport_with_data(client_app):
    client, app_module = client_app
    airport = app_module.VALID_AIRPORTS[0]

    app_module.df_preds[airport] = [
        {"timestamp": "2025-01-01T00:00:00", "mean": 1, "q30": 0.8, "q70": 1.2}
    ]

    resp = client.get(f"/forecast/{airport}")
    assert resp.status_code == 200
    assert resp.get_json()["predictions"] == app_module.df_preds[airport]


def test_metrics_endpoint_returns_metrics(client_app):
    client, app_module = client_app

    app_module.train_metrics["CPH"] = {"total_time_seconds": 1.23}

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.get_json()["CPH"]["total_time_seconds"] == 1.23
