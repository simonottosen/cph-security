"""Smoke-test that ``project.app``'s ``__main__`` block boots without crashing.

The module is executed in a sub-process with the external seams stubbed *before*
import so the test stays fast and offline:

* **requests.get** – returns a tiny synthetic payload (so ``_load_raw`` succeeds).
* **apscheduler ... BackgroundScheduler** – a no-op so ``_initialize`` schedules
  nothing and returns immediately.
* **flask.Flask.run** – returns instead of blocking on the serving loop.

Chronos/torch are absent in the test image, so ``app`` falls back to its naive
pipeline automatically — no stub needed. The sub-process must exit cleanly (0).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.slow
def test_project_app_main_starts_quickly():
    helper_script = textwrap.dedent(
        r"""
        import sys, types

        # --- Stub requests so _load_raw() returns a tiny frame offline. -------
        requests = types.ModuleType("requests")

        class _Resp:
            def raise_for_status(self): pass
            def json(self):
                return [{"timestamp": "2024-01-01T00:00:00", "airport": "CPH", "queue": 1}]

        requests.get = lambda *a, **kw: _Resp()
        class _ConnErr(Exception):
            pass
        requests.ConnectionError = _ConnErr
        sys.modules["requests"] = requests

        # --- Stub APScheduler so the scheduler start is a no-op. --------------
        sched_pkg = types.ModuleType("apscheduler")
        sched_sub = types.ModuleType("apscheduler.schedulers")
        sched_bg = types.ModuleType("apscheduler.schedulers.background")

        class _DummyScheduler:
            def __init__(self, *a, **kw): pass
            def add_job(self, *a, **kw): pass
            def start(self): pass

        sched_bg.BackgroundScheduler = _DummyScheduler
        sched_pkg.schedulers = sched_sub
        sched_sub.background = sched_bg
        sys.modules["apscheduler"] = sched_pkg
        sys.modules["apscheduler.schedulers"] = sched_sub
        sys.modules["apscheduler.schedulers.background"] = sched_bg

        # --- Don't block on the serving loop. --------------------------------
        import flask
        flask.Flask.run = lambda *a, **kw: None

        import runpy
        runpy.run_module("project.app", run_name="__main__")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", helper_script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, (
        f"Sub-process exited with {completed.returncode}\nSTDERR:\n{completed.stderr}"
    )
