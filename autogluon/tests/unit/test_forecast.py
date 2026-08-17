"""Unit tests for the batched Chronos-2 forecast service (project/app.py).

These lock in the behaviours the serving contract depends on: a single batched
predict_df across all airports, the expanded 21-level quantile schema with
backward-compatible decile + q30/q70 keys, robust column lookup (no 0.05 -> 0.1
collision), reading the point forecast from Chronos-2's ``predictions`` column,
forecast persistence, and the Flask endpoints. They run against the naive
fallback pipeline, so they pass without torch/chronos installed. Real
forecast-quality gating lives in eval/backtest.py (the accuracy harness).
"""
import contextlib
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


def _seed_forecast(code="CPH", ends_in_minutes=240):
    """Point ``df_preds`` at a single forecast ending the given distance from now."""
    ts = pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(minutes=ends_in_minutes)
    with app._state_lock:
        app.df_preds.clear()
        app.train_metrics.clear()
        app.df_preds[code] = [{
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "mean": 5.0,
            "q30": 4.0,
            "q70": 6.0,
        }]
        app.train_metrics[code] = {"total_time_seconds": 1.2, "status": "ok"}


def test_flask_endpoints():
    client = app.app.test_client()

    # Seed state directly (retrain rebinds these globals atomically at runtime).
    _seed_forecast("CPH", ends_in_minutes=240)

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
    assert body["airports_usable"] == 1
    assert body["airports"]["CPH"]["has_forecast"] is True


def test_health_reports_unhealthy_when_the_horizon_has_elapsed():
    """The regression that went unnoticed for a week: serving an elapsed window.

    /health answered "ok" purely because forecasts were loaded, so an entirely
    past horizon looked healthy while every consumer discarded it.
    """
    client = app.app.test_client()
    _seed_forecast("CPH", ends_in_minutes=-120)

    h = client.get("/health")
    assert h.status_code == 503, "an elapsed horizon must not report healthy"
    body = h.get_json()
    assert body["status"] == "stale"
    assert body["airports_usable"] == 0
    # Still loaded, which is exactly why counting them proved nothing.
    assert body["airports_loaded"] == 1
    assert body["airports"]["CPH"]["horizon_remaining_minutes"] < 0

    # And it recovers once a fresh forecast lands.
    _seed_forecast("CPH", ends_in_minutes=240)
    assert client.get("/health").status_code == 200


def test_health_tolerates_a_single_dead_feed():
    """One airport with no data must not mark the whole service unhealthy."""
    client = app.app.test_client()
    _seed_forecast("CPH", ends_in_minutes=240)
    with app._state_lock:
        # EDI has produced nothing since 2025-11-21; it reports a status, not a mean.
        app.df_preds["EDI"] = [{"timestamp": "2026-01-01T00:00:00", "message": "EDI: no rows"}]
        app.train_metrics["EDI"] = {"status": "no_rows"}

    h = client.get("/health")
    assert h.status_code == 200
    body = h.get_json()
    assert body["airports"]["EDI"]["has_forecast"] is False
    assert body["airports"]["EDI"]["status"] == "no_rows"


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


def _raw_with_gap(code="CPH"):
    """History split by an outage: a long old segment, a shorter recent one.

    Mirrors real CPH data on 2026-08-17, where a 14 h outage left a 51-day
    stretch ending 9 days ago alongside a 2.5-day stretch ending now.
    """
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("5min")
    old = pd.date_range(now - pd.Timedelta(days=20), now - pd.Timedelta(days=9), freq="5min")
    recent = pd.date_range(now - pd.Timedelta(days=2), now, freq="5min")
    idx = old.append(recent)
    vals = 6.0 + 3.0 * np.sin(np.arange(len(idx)) / 40.0)
    return pd.DataFrame({
        "airport": code,
        "timestamp": idx.astype(str),
        "queue": np.maximum(0.0, vals),
    }), old[-1], recent[-1]


def test_context_anchors_to_most_recent_segment_not_longest():
    """A longer but older segment must not win: it forecasts an elapsed window."""
    df_raw, old_end, recent_end = _raw_with_gap("CPH")
    ctx, stats, _ = app._prepare_airport_context(df_raw, "CPH")

    assert ctx is not None, f"expected a usable context, got status={stats['status']}"
    assert stats["status"] == "ok"

    # Compare on the resample grid: the raw samples are floored into it.
    expected_end = recent_end.floor(app.RESAMPLE_FREQUENCY)
    selected_end = pd.Timestamp(stats["selected_segment_end"])
    assert selected_end == expected_end, (
        f"anchored to {selected_end}, expected the recent segment end {expected_end}"
    )
    # The regression being locked in: the old segment is far longer.
    assert selected_end > old_end
    assert ctx["timestamp"].max() == expected_end


def test_stale_history_is_refused_rather_than_forecast_into_the_past():
    """Context older than MAX_CONTEXT_AGE_MINUTES yields a status, not a forecast."""
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("5min")
    # One segment only, ending well beyond the staleness limit.
    end = now - pd.Timedelta(minutes=app.MAX_CONTEXT_AGE_MINUTES + 120)
    idx = pd.date_range(end - pd.Timedelta(days=5), end, freq="5min")
    vals = 6.0 + 3.0 * np.sin(np.arange(len(idx)) / 40.0)
    df_raw = pd.DataFrame({
        "airport": "CPH",
        "timestamp": idx.astype(str),
        "queue": np.maximum(0.0, vals),
    })

    ctx, stats, _ = app._prepare_airport_context(df_raw, "CPH")
    assert ctx is None, "a cold context must not produce a forecast"
    assert stats["status"] == "stale_history"
    assert stats["context_age_minutes"] >= app.MAX_CONTEXT_AGE_MINUTES


def _diurnal_raw(code="CPH", days=20, gap=None, freq=None):
    """A strongly diurnal series, optionally with a hole punched in it.

    Returns ``(raw_frame, truth_series)`` so a test can compare a reconstruction
    against the values that were withheld.
    """
    freq = freq or app.RESAMPLE_FREQUENCY
    now = pd.Timestamp.now("UTC").tz_localize(None).floor(freq)
    idx = pd.date_range(now - pd.Timedelta(days=days), now, freq=freq, name="timestamp")
    minute_of_day = idx.hour * 60 + idx.minute
    truth = pd.Series(12.0 + 9.0 * np.sin(2 * np.pi * minute_of_day / 1440.0), index=idx)

    keep = pd.Series(True, index=idx)
    if gap is not None:
        keep.loc[gap[0]:gap[1]] = False
    observed = truth[keep]
    raw = pd.DataFrame({
        "airport": code,
        "timestamp": observed.index.astype(str),
        "queue": observed.to_numpy(),
    })
    return raw, truth


def test_long_gap_is_bridged_and_keeps_the_daily_shape():
    """A multi-hour outage must not truncate the context nor flatten the cycle.

    The old behaviour discarded every observation before an unbridged gap, which
    turned 60 days of history into 2.6 usable days. Linear interpolation would
    keep the rows but replace a whole night-and-morning cycle with a ramp.
    """
    now = pd.Timestamp.now("UTC").tz_localize(None).floor(app.RESAMPLE_FREQUENCY)
    step = pd.Timedelta(app.RESAMPLE_FREQUENCY)
    gap_start = (now - pd.Timedelta(days=5)).floor("h")
    gap_end = gap_start + pd.Timedelta(hours=12)
    raw, truth = _diurnal_raw(gap=(gap_start, gap_end))

    ctx, stats, _ = app._prepare_airport_context(raw, "CPH")
    assert ctx is not None, f"gap should be bridged, got status={stats['status']}"
    assert stats["status"] == "ok"

    # The whole span survives: one segment, not a post-gap remnant.
    assert pd.Timestamp(stats["selected_segment_start"]) == truth.index.min()
    assert pd.Timestamp(stats["selected_segment_end"]) == truth.index.max()
    # The hole spans gap_start..gap_end inclusive; its anchors sit one step outside.
    hole = truth.loc[gap_start:gap_end].index
    assert stats["rows_seasonally_filled"] == len(hole)
    assert stats["longest_gap_filled_steps"] == len(hole)

    filled = ctx.set_index("timestamp")["queue"]
    inside = filled.loc[hole]
    actual = truth.loc[hole]

    # A straight line between the surviving observations either side of the hole:
    # what plain interpolation would have produced.
    left, right = gap_start - step, gap_end + step
    linear = pd.Series(
        np.linspace(truth.loc[left], truth.loc[right], len(hole) + 2)[1:-1],
        index=hole,
    )
    seasonal_rmse = float(np.sqrt(((inside - actual) ** 2).mean()))
    linear_rmse = float(np.sqrt(((linear - actual) ** 2).mean()))
    assert seasonal_rmse < linear_rmse / 2, (
        f"seasonal fill rmse {seasonal_rmse:.3f} should clearly beat linear {linear_rmse:.3f}"
    )
    # The reconstruction spans a real daily swing rather than sitting flat.
    assert inside.max() - inside.min() > 0.5 * (actual.max() - actual.min())


def test_gap_longer_than_the_limit_still_splits_the_series():
    """We reconstruct outages, not multi-day silence."""
    now = pd.Timestamp.now("UTC").tz_localize(None).floor(app.RESAMPLE_FREQUENCY)
    step = pd.Timedelta(app.RESAMPLE_FREQUENCY)
    over_limit = step * (app.MAX_FILL_GAP_STEPS + 10)
    gap_end = now - pd.Timedelta(days=3)
    raw, _ = _diurnal_raw(days=30, gap=(gap_end - over_limit, gap_end))

    ctx, stats, _ = app._prepare_airport_context(raw, "CPH")
    assert ctx is not None
    # Context begins after the gap, so the unbridgeable hole cost us the earlier data.
    assert pd.Timestamp(stats["selected_segment_start"]) >= gap_end
    assert stats["rows_missing_after_fill"] > 0


def test_fill_never_extrapolates_past_the_observations():
    """Leading and trailing holes have no anchor, so they must stay missing."""
    _, truth = _diurnal_raw(days=10)
    series = truth.copy()
    series.iloc[:5] = np.nan
    series.iloc[-5:] = np.nan
    cov = app._build_historical_covariate_tables(truth.to_frame("queue"))

    filled, rows_filled, _ = app._seasonal_fill(series, cov, app.MAX_FILL_GAP_STEPS)
    assert rows_filled == 0
    assert filled.iloc[:5].isna().all()
    assert filled.iloc[-5:].isna().all()


def test_reconstruction_is_never_negative():
    """A downward drift must not invent a negative queue length."""
    _, truth = _diurnal_raw(days=10)
    series = truth.copy()
    # Bracket a low-season hole with near-zero observations so the residual drift
    # pulls the seasonal profile below zero.
    hole = slice(100, 100 + min(20, app.MAX_FILL_GAP_STEPS))
    series.iloc[hole] = np.nan
    series.iloc[hole.start - 1] = 0.0
    series.iloc[hole.stop] = 0.0
    cov = app._build_historical_covariate_tables(truth.to_frame("queue"))

    filled, rows_filled, _ = app._seasonal_fill(series, cov, app.MAX_FILL_GAP_STEPS)
    assert rows_filled > 0
    assert (filled.dropna() >= 0).all(), f"negative values: {filled[filled < 0].tolist()}"


def _copy_on_write():
    """Force pandas' copy-on-write semantics, which are unconditional from pandas 3.

    pandas 3 removed the toggle, so asking for it there raises; that version already
    behaves this way, hence the no-op fallback.
    """
    try:
        with pd.option_context("mode.copy_on_write", True):
            pass
    except Exception:
        return contextlib.nullcontext()
    return pd.option_context("mode.copy_on_write", True)


def test_fill_works_under_copy_on_write():
    """Regression: the fill must not try to mutate a copy-on-write view.

    Under CoW, ``Series.to_numpy()`` hands back a read-only view when the dtype
    already matches, so writing to it raises ValueError -- which broke every airport
    that had any gap at all. It passed locally on pandas 2 and failed in CI on
    pandas 3, so force the strict semantics here rather than inheriting whichever
    version happens to be installed.
    """
    _, truth = _diurnal_raw(days=10)
    cov = app._build_historical_covariate_tables(truth.to_frame("queue"))

    with _copy_on_write():
        series = truth.copy()
        series.iloc[100:110] = np.nan
        filled, rows_filled, _ = app._seasonal_fill(series, cov, app.MAX_FILL_GAP_STEPS)

    assert rows_filled == 10
    assert filled.notna().all()
    # The input must be left exactly as it was found.
    assert series.iloc[100:110].isna().all()


def test_negative_sentinel_readings_are_treated_as_missing():
    """LHR reports -1 overnight for 'no reading'; -1 minutes is not a queue."""
    raw, truth = _diurnal_raw(days=10)
    sentinel_at = raw.index[len(raw) // 2]
    raw.loc[sentinel_at, "queue"] = -1

    ctx, stats, _ = app._prepare_airport_context(raw, "CPH")
    assert ctx is not None
    assert stats["negative_sentinel_rows"] == 1
    assert (ctx["queue"] >= 0).all()
    # It became a one-step hole that the fill bridged, not a real -1 observation.
    assert -1 not in ctx["queue"].to_numpy()


def test_default_horizon_stays_eight_hours():
    """Guard the coupling between grid and horizon.

    PREDICTION_LENGTH counts steps, so changing RESAMPLE_FREQUENCY alone silently
    rescales the published forecast window.
    """
    horizon = pd.Timedelta(app.RESAMPLE_FREQUENCY) * app.PREDICTION_LENGTH
    assert horizon == pd.Timedelta(hours=8), f"horizon drifted to {horizon}"


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
