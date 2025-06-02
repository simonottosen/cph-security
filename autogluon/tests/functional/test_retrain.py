

"""
Functional tests for the high‑level **retrain()** routine in *project.app*.

We monkey‑patch heavy or side‑effectful dependencies to make the test fast,
deterministic and independent of external systems:

* **requests.get**           → returns a tiny synthetic JSON payload.
* **_train_airport**         → quick stub that fabricates predictions/metrics.
* **ProcessPoolExecutor**    → executes synchronously in‑process.
* **as_completed**           → returns futures immediately (they are done).

Two scenarios are covered:

1. **Happy path** — verifies that forecasts and metrics are collected for
   *all* airports and stored in the expected global dicts.
2. **Network failure** — `requests.get` raises `ConnectionError`; the
   exception must propagate so the caller (e.g., scheduler) can handle it.
"""

from __future__ import annotations

import types
from typing import Dict, List

import pytest


# --------------------------------------------------------------------------- #
#                         Shared dummy implementations                        #
# --------------------------------------------------------------------------- #
def _make_requests_stub(valid_airports: List[str]):
    """Return a stub `get()` that produces one JSON row per airport."""
    class _Resp:
        def raise_for_status(self):  # noqa: D401
            pass

        def json(self) -> List[Dict]:
            return [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "airport": code,
                    "queue": 1,
                }
                for code in valid_airports
            ]

    return lambda *a, **kw: _Resp()


def _dummy_train_airport(code: str, df_raw):
    """Fast stub for `project.app._train_airport`."""
    preds = [
        {
            "timestamp": "2025-01-01T00:00:00",
            "mean": 1.0,
            "q30": 0.8,
            "q70": 1.2,
        }
    ]
    metrics = {"dummy_metric": 123}
    return code, preds, metrics


class _ImmediateFuture:
    """
    Minimal stand‑in for `concurrent.futures.Future`.

    * Executes the callable lazily – on **first** `result()` – so that
      exceptions propagate at the correct time (mirrors real futures).
    """
    def __init__(self, fn, *a, **kw):
        self._fn = fn
        self._args = a
        self._kw = kw
        self._executed = False
        self._result = None
        self._exc = None

    # ------------------------------------------------------------------ #
    def _execute_once(self):
        if self._executed:
            return
        try:
            self._result = self._fn(*self._args, **self._kw)
        except Exception as e:  # noqa: BLE001
            self._exc = e
        self._executed = True

    def result(self, *_, **__):
        self._execute_once()
        if self._exc:
            raise self._exc
        return self._result


class _ImmediateExecutor:
    """Drop‑in replacement for `concurrent.futures.ProcessPoolExecutor`."""
    def __init__(self, *_, **__):
        # Accept but ignore any constructor arguments such as *max_workers*.
        pass

    def __enter__(self, *_):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *a, **kw):
        return _ImmediateFuture(fn, *a, **kw)


def _immediate_as_completed(fs, *_, **__):
    """Return the futures immediately – they’re already complete."""
    return fs


# --------------------------------------------------------------------------- #
#                              Test: happy path                               #
# --------------------------------------------------------------------------- #
def test_retrain_merges_results(monkeypatch):
    import importlib

    app = importlib.import_module("project.app")

    # --- Patch external dependencies ------------------------------------ #
    # 1) requests.get
    monkeypatch.setattr(
        app.requests,
        "get",
        _make_requests_stub(app.VALID_AIRPORTS),
        raising=True,
    )
    # 2) _train_airport
    monkeypatch.setattr(app, "_train_airport", _dummy_train_airport, raising=True)
    # 3) Executor + as_completed
    monkeypatch.setattr(app, "ProcessPoolExecutor", _ImmediateExecutor, raising=True)
    monkeypatch.setattr(app, "as_completed", _immediate_as_completed, raising=True)

    # Clear any residual state
    app.df_preds.clear()
    app.train_metrics.clear()

    # ----------------------------- Act ----------------------------------- #
    app.retrain()

    # --------------------------- Assertions ------------------------------ #
    assert set(app.df_preds) == set(app.VALID_AIRPORTS)
    assert set(app.train_metrics) == set(app.VALID_AIRPORTS)

    # Verify content integrity for one sample airport.
    sample_code = app.VALID_AIRPORTS[0]
    assert app.df_preds[sample_code][0]["mean"] == 1.0
    assert app.train_metrics[sample_code]["dummy_metric"] == 123


# --------------------------------------------------------------------------- #
#                       Test: network failure propagation                     #
# --------------------------------------------------------------------------- #
def test_retrain_raises_on_connection_error(monkeypatch):
    import importlib

    app = importlib.import_module("project.app")

    # Patch requests.get to raise ConnectionError
    def _raise(*a, **kw):
        raise app.requests.ConnectionError("boom")

    monkeypatch.setattr(app.requests, "get", _raise, raising=True)

    with pytest.raises(app.requests.ConnectionError):
        app.retrain()