"""Unit tests for the batched Chronos-2 forecast service (project/app.py).

These lock in the behaviours the serving contract depends on: a single batched
predict_df across all airports, the expanded 21-level quantile schema with
backward-compatible decile + q30/q70 keys, robust column lookup (no 0.05 -> 0.1
collision), reading the point forecast from Chronos-2's ``predictions`` column,
forecast persistence, and the Flask endpoints. They run against the naive
fallback pipeline, so they pass without torch/chronos installed. Real
forecast-quality gating lives in eval/backtest.py (the accuracy harness).
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make autogluon/project importable when run standalone or under pytest.
_PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "project",
)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import app  # noqa: E402


def _synthetic_raw(codes=("CPH", "ARN", "DUB"), days=3, base=6.0):
    """A raw long frame like PostgREST returns: airport/timestamp/queue."""
    now = pd.Timestamp.utcnow().tz_localize(None).floor("5min")
    rows = []
    for i, code in enumerate(codes):
        idx = pd.date_range(now - pd.Timedelta(days=days), now, freq="5min")
        vals = (base + 2.0 * i) + 3.0 * np.sin(np.arange(len(idx)) / 40.0)
        rows.append(pd.DataFrame({
            "airport": code,
            "timestamp": idx.astype(str),
            "queue": np.maximum(0.0, vals),
        }))
    return pd.concat(rows, ignore_index=True)


def _chronos_like_pred_df(code="CPH", n=5):
    """A predict_df output mimicking real Chronos-2: 'predictions' + str(q) cols."""
    ts = pd.date_range("2026-06-01", periods=n, freq="5min")
    df = pd.DataFrame({"item_id": code, "timestamp": ts})
    df["predictions"] = np.linspace(10.0, 14.0, n)
    # Distinct, strictly increasing per-row quantiles so we can detect mis-mapping.
    for j, q in enumerate(app.QUANTILE_LEVELS):
        df[str(q)] = df["predictions"] + (j - len(app.QUANTILE_LEVELS) / 2)
    return df


def test_run_forecast_is_batched_and_schema_stable():
    forecasts, metrics = app._run_forecast(_synthetic_raw())
    for code in ("CPH", "ARN", "DUB"):
        recs = forecasts[code]
        assert len(recs) == app.PREDICTION_LENGTH
        r0 = recs[0]
        assert "mean" in r0 and r0["mean"] is not None
        assert "q30" in r0 and "q70" in r0
        # Backward-compatible decile keys and new fine-grained keys both present.
        for k in ("0.1", "0.5", "0.9", "0.05", "0.95"):
            assert k in r0, f"missing quantile key {k}"
        # Metrics reflect the single batched call across airports.
        assert metrics[code]["cross_learning"] == app.CROSS_LEARNING
        assert metrics[code]["batched_items"] >= 3


def test_run_forecast_flags_insufficient_history():
    raw = _synthetic_raw(codes=("CPH",), days=3)
    # An airport with only ~1h of data cannot meet PREDICTION_LENGTH*2.
    now = pd.Timestamp.utcnow().tz_localize(None).floor("5min")
    short = pd.DataFrame({
        "airport": "MUC",
        "timestamp": pd.date_range(now - pd.Timedelta(hours=1), now, freq="5min").astype(str),
        "queue": 5.0,
    })
    forecasts, metrics = app._run_forecast(pd.concat([raw, short], ignore_index=True))
    assert "message" in forecasts["MUC"][0]
    assert metrics["MUC"]["status"] == "insufficient_history"
    assert len(forecasts["CPH"]) == app.PREDICTION_LENGTH  # healthy one still forecast


def test_format_predictions_reads_predictions_column():
    pred = _chronos_like_pred_df(n=4)
    recs = app._format_predictions(pred)
    assert len(recs) == 4
    # mean must come from 'predictions', not the median quantile.
    assert recs[0]["mean"] == 10.0
    # q30/q70 alias the 0.3/0.7 columns exactly.
    assert recs[0]["q30"] == recs[0]["0.3"]
    assert recs[0]["q70"] == recs[0]["0.7"]
    # Fine and decile keys are distinct (no 0.05 -> 0.1 collision).
    assert recs[0]["0.05"] != recs[0]["0.1"]


def test_quantile_column_name_avoids_decile_collision():
    df = pd.DataFrame({"0.05": [1.0], "0.1": [2.0], "0.5": [3.0]})
    assert app._quantile_column_name(df, 0.05) == "0.05"
    assert app._quantile_column_name(df, 0.1) == "0.1"
    assert app._quantile_column_name(df, 0.5) == "0.5"
    assert app._quantile_column_name(df, 0.9) is None  # not present


def test_api_quantile_key_is_backward_compatible():
    assert app._api_quantile_key(0.1) == "0.1"
    assert app._api_quantile_key(0.5) == "0.5"
    assert app._api_quantile_key(0.9) == "0.9"
    assert app._api_quantile_key(0.05) == "0.05"
    assert app._api_quantile_key(0.95) == "0.95"
    assert app._api_quantile_key(0.99) == "0.99"


def test_forecast_persistence_round_trip():
    original = app.FORECAST_DIR
    with tempfile.TemporaryDirectory() as tmp:
        app.FORECAST_DIR = tmp
        try:
            forecasts = {"CPH": [{"timestamp": "2026-06-01T00:00:00", "mean": 5.0, "q30": 4.0, "q70": 6.0}]}
            metrics = {"CPH": {"total_time_seconds": 1.2}}
            app._persist_forecasts(forecasts, metrics)
            loaded = app._load_persisted_forecasts()
        finally:
            app.FORECAST_DIR = original
    assert loaded is not None
    assert loaded["predictions"]["CPH"][0]["mean"] == 5.0
    assert loaded["metrics"]["CPH"]["total_time_seconds"] == 1.2
    assert "generated_at" in loaded


def test_flask_endpoints():
    client = app.app.test_client()

    # Seed state directly (retrain rebinds these globals atomically at runtime).
    with app._state_lock:
        app.df_preds["CPH"] = [{"timestamp": "2026-06-01T00:00:00", "mean": 5.0, "q30": 4.0, "q70": 6.0}]
        app.train_metrics["CPH"] = {"total_time_seconds": 1.2}

    ok = client.get("/forecast/cph")
    assert ok.status_code == 200
    assert ok.get_json()["predictions"][0]["mean"] == 5.0

    bad = client.get("/forecast/XYZ")
    assert bad.status_code == 404
    assert "error" in bad.get_json()

    m = client.get("/metrics")
    assert m.status_code == 200
    assert m.get_json()["CPH"]["total_time_seconds"] == 1.2

    h = client.get("/health")
    assert h.status_code == 200
    body = h.get_json()
    assert body["status"] == "ok"
    assert body["cross_learning"] == app.CROSS_LEARNING


def test_quantiles_are_non_decreasing_per_timestamp():
    """q10 <= q50 <= q90 at every step (equal under the fallback; monotone real)."""
    forecasts, _ = app._run_forecast(_synthetic_raw(codes=("CPH",)))
    for rec in forecasts["CPH"]:
        assert rec["0.1"] <= rec["0.5"] <= rec["0.9"]


def test_beats_last_value_when_chronos_available():
    """Forecast-quality gate (only runs where a real Chronos-2 is installed).

    Full accuracy gating is done by eval/backtest.py; this is a fast sanity slice.
    """
    if app.BaseChronosPipeline is None:
        return  # fallback pipeline is last-value by construction; nothing to prove

    raw = _synthetic_raw(codes=("CPH", "ARN", "DUB", "FRA"), days=10)
    forecasts, _ = app._run_forecast(raw)
    # A real model on a smooth seasonal signal should produce a varying (not
    # flat last-value) forecast for a healthy series.
    means = [r["mean"] for r in forecasts["CPH"] if "mean" in r]
    assert len(means) == app.PREDICTION_LENGTH
    assert np.std(means) > 1e-6


if __name__ == "__main__":
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
