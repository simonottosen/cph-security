

"""
Smoke‑test that the **project.app** `__main__` block starts up without
crashing.

To keep the test < 2 s and independent of external APIs / heavy ML code,
we run the module in a **sub‑process** with a handful of strategic
monkey‑patches applied *before* import:

* **autogluon.timeseries** – replaced by a dummy predictor + dataframe.
* **requests.get** – returns a tiny synthetic payload.
* **concurrent.futures.ProcessPoolExecutor** – executes jobs synchronously.
* **apscheduler.schedulers.background.BackgroundScheduler** – no‑op.
* **flask.Flask.run** – returns immediately instead of blocking.

The sub‑process is expected to exit cleanly (code 0) well before the
pytest timeout.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.slow
def test_project_app_main_starts_quickly():
    """Run `python -m project.app` with heavy dependencies stubbed out."""
    helper_script = textwrap.dedent(
        r"""
        import sys, types, runpy, datetime

        # ------------------------------------------------------------------ #
        #                       Stub *autogluon.timeseries*                  #
        # ------------------------------------------------------------------ #
        ag = types.ModuleType("autogluon")
        ts = types.ModuleType("autogluon.timeseries")

        class _DummyTSDF(dict):
            @classmethod
            def from_data_frame(cls, df, *_, **__):
                return cls({"dummy": df})

            # Production code calls train_test_split(prediction_length=…)
            def train_test_split(self, *_, **__):
                return self, self

        class _DummyPred:
            def __init__(self, *_, **__): pass

            def fit(self, *_, **__):  # noqa: D401
                return self

            def predict(self, *_, **__):
                import pandas as pd, numpy as np
                idx = pd.date_range("2024-01-01", periods=96, freq="5min")
                return pd.DataFrame(
                    {
                        "mean": np.zeros(len(idx)),
                        "0.3": np.zeros(len(idx)),
                        "0.7": np.zeros(len(idx)),
                    },
                    index=idx,
                )

            def fit_summary(self, *_, **__):
                return {"model_performance": {"Dummy": {"MAPE": 0.1}}}

            def persist(self, *_, **__): pass

        ts.TimeSeriesDataFrame = _DummyTSDF
        ts.TimeSeriesPredictor = _DummyPred
        ag.timeseries = ts
        sys.modules["autogluon"] = ag
        sys.modules["autogluon.timeseries"] = ts

        # ------------------------------------------------------------------ #
        #                         Stub *requests*                            #
        # ------------------------------------------------------------------ #
        import json, datetime as _dt

        requests = types.ModuleType("requests")

        class _Resp:  # minimal Response stand‑in
            def raise_for_status(self): pass

            def json(self):
                # One row with plausible keys expected by _train_airport
                return [
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "airport": "CPH",
                        "queue": 1,
                    }
                ]

        requests.get = lambda *a, **kw: _Resp()
        sys.modules["requests"] = requests

        # ------------------------------------------------------------------ #
        #                 Stub concurrent.futures Executor                   #
        # ------------------------------------------------------------------ #
        import concurrent.futures as _cf

        class _ImmediateFuture:
            """
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

        class _ImmediateExec:
            def __init__(self, *_, **__):  # Accept max_workers or other kwargs.
                pass

            def __enter__(self, *_): return self

            def __exit__(self, *exc): pass

            def submit(self, fn, *a, **kw):
                return _ImmediateFuture(fn, *a, **kw)

        def _as_completed(fs, *_, **__):
            # Just return the list itself – they're ready immediately.
            return fs

        _cf.ProcessPoolExecutor = _ImmediateExec
        _cf.as_completed = _as_completed

        # ------------------------------------------------------------------ #
        #                      Stub APScheduler scheduler                     #
        # ------------------------------------------------------------------ #
        sched_pkg = types.ModuleType("apscheduler")
        sched_sub = types.ModuleType("apscheduler.schedulers")
        sched_bg = types.ModuleType("apscheduler.schedulers.background")

        class _DummyScheduler:
            def __init__(self, *a, **kw): pass
            def add_job(self, *a, **kw): pass
            def start(self): pass

        sched_bg.BackgroundScheduler = _DummyScheduler
        sched_pkg.schedulers = sched_sub
        sys.modules["apscheduler"] = sched_pkg
        sys.modules["apscheduler.schedulers"] = sched_sub
        sys.modules["apscheduler.schedulers.background"] = sched_bg

        # ------------------------------------------------------------------ #
        #                Patch flask.Flask.run to return quickly              #
        # ------------------------------------------------------------------ #
        import flask
        flask.Flask.run = lambda *a, **kw: None

        # Finally: execute the module as a script.
        runpy.run_module("project.app", run_name="__main__")
        """
    )

    # Run the helper in a separate interpreter to isolate side‑effects.
    completed = subprocess.run(
        [sys.executable, "-c", helper_script],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert (
        completed.returncode == 0
    ), f"Sub‑process exited with {completed.returncode}\nSTDERR:\n{completed.stderr}"