"""Tests for the history fetch — the query it sends and the truncation it refuses.

``_load_raw`` used to ask PostgREST for the entire table on every refresh: 2.85 M
rows and ~240 MB an hour, where the forecaster reads three columns and the last ~14
months. These tests pin down the two properties that keep that honest:

* the request asks only for the columns and the window that are actually used, and
  the window is wide enough for the longest covariate lag; and
* a response the API has silently capped is rejected rather than forecast from.

``requests.get`` is stubbed throughout, so nothing here touches the network.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


@pytest.fixture
def app():
    return importlib.import_module("project.app")


class _Resp:
    """Minimal stand-in for a requests Response, including the headers we read."""

    def __init__(self, rows, total=None, headers=None):
        self._rows = rows
        if headers is not None:
            self.headers = headers
        else:
            end = max(len(rows) - 1, 0)
            total = len(rows) if total is None else total
            self.headers = {"Content-Range": f"0-{end}/{total}"}

    def raise_for_status(self):
        pass

    def json(self):
        return self._rows


def _rows(n=3):
    return [
        {"airport": "CPH", "timestamp": f"2026-01-01T00:0{i}:00", "queue": float(i)}
        for i in range(n)
    ]


def _capture(app, monkeypatch, response):
    """Run _load_raw against a stubbed requests.get and return the captured call."""
    seen = {}

    def _get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params or {}, headers=headers or {}, timeout=timeout)
        return response

    monkeypatch.setattr(app.requests, "get", _get, raising=True)
    df = app._load_raw()
    return seen, df


def test_history_window_covers_the_longest_covariate_lag(app):
    """The fetch must outlast the oldest lookup done against it.

    queue_lag_365d reads the history 365 days before each context row, and the
    oldest context row is CONTEXT_DAYS back. A window shorter than the sum does not
    fail -- the lag just resolves to nothing and falls back to the seasonal profile
    for every row -- so nothing would report it. Hence the assertion.
    """
    assert app.HISTORY_DAYS >= app.CONTEXT_DAYS + app.MAX_COVARIATE_LAG_DAYS, (
        f"HISTORY_DAYS={app.HISTORY_DAYS} cannot serve a {app.CONTEXT_DAYS}-day context "
        f"with a {app.MAX_COVARIATE_LAG_DAYS}-day lag"
    )


def test_request_asks_only_for_the_columns_and_window_used(app, monkeypatch):
    seen, df = _capture(app, monkeypatch, _Resp(_rows()))

    assert seen["params"]["select"] == "airport,timestamp,queue"
    # Whatever the forecaster reads must be in the projection, or it arrives empty.
    assert set(app.RAW_COLUMNS) == {"airport", "timestamp", "queue"}

    gte = seen["params"]["timestamp"]
    assert gte.startswith("gte."), gte
    cutoff = pd.Timestamp(gte.removeprefix("gte."))
    expected = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=app.HISTORY_DAYS)
    # Generous tolerance: this pins the window, not the clock.
    assert abs((cutoff - expected).total_seconds()) < 300, f"cutoff {cutoff} vs {expected}"

    # Without count=exact PostgREST reports "/*" and truncation is undetectable.
    assert seen["headers"].get("Prefer") == "count=exact"
    assert seen["timeout"] == app.FETCH_TIMEOUT
    assert len(df) == 3


def test_truncated_history_is_rejected(app, monkeypatch):
    """A capped response must raise rather than become a quietly worse forecast.

    The query is unordered, so the rows that survive a cap are arbitrary. Forecasting
    from them would look completely normal -- same row count in the logs, same shape
    of output -- while resting on a fraction of the history.
    """
    with pytest.raises(RuntimeError, match="truncated"):
        _capture(app, monkeypatch, _Resp(_rows(3), total=940018))


def test_complete_history_is_accepted(app, monkeypatch):
    """The guard must not fire when the row count matches the reported total."""
    _, df = _capture(app, monkeypatch, _Resp(_rows(5), total=5))
    assert len(df) == 5


def test_empty_history_is_not_mistaken_for_truncation(app, monkeypatch):
    """PostgREST answers "*/0" for an empty result; 0 of 0 is complete, not capped."""
    _, df = _capture(app, monkeypatch, _Resp([], headers={"Content-Range": "*/0"}))
    assert df.empty


def test_missing_content_range_is_tolerated(app, monkeypatch):
    """Not every proxy forwards the header; absence must not break the fetch.

    count=exact is a request, not a guarantee, so the guard has to degrade to letting
    the response through rather than refusing to forecast at all.
    """
    _, df = _capture(app, monkeypatch, _Resp(_rows(4), headers={}))
    assert len(df) == 4

    _, df = _capture(app, monkeypatch, _Resp(_rows(4), headers={"Content-Range": "0-3/*"}))
    assert len(df) == 4


def test_retrain_keeps_last_good_when_history_is_truncated(app, monkeypatch, tmp_path):
    """The raise has to land as a skipped refresh, not a crashed scheduler job."""
    monkeypatch.setattr(app, "FORECAST_DIR", str(tmp_path), raising=True)
    with app._state_lock:
        app.df_preds = {"CPH": [{"timestamp": "2026-01-01T00:00:00", "mean": 9.0}]}
        app.train_metrics = {"CPH": {"cross_learning": True}}
    last_good = app.df_preds

    def _get(url, params=None, headers=None, timeout=None):
        return _Resp(_rows(3), total=940018)

    monkeypatch.setattr(app.requests, "get", _get, raising=True)

    app.retrain()  # must not raise

    assert app.df_preds is last_good
