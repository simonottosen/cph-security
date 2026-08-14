"""Tests for the backtest harness metric math (MAE / RMSE / WQL)."""
import os
import sys

import numpy as np
import pandas as pd

# Make eval/backtest.py importable when run standalone (mirrors conftest.py).
_EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import backtest  # noqa: E402


def test_quantile_loss_penalises_asymmetrically():
    y = np.array([10.0])
    # Under-prediction (y > pred): high quantile should be penalised more.
    under_p90 = backtest.quantile_loss(y, np.array([6.0]), 0.9)[0]
    under_p10 = backtest.quantile_loss(y, np.array([6.0]), 0.1)[0]
    assert under_p90 > under_p10
    # Over-prediction (y < pred): low quantile penalised more.
    over_p10 = backtest.quantile_loss(np.array([10.0]), np.array([14.0]), 0.1)[0]
    over_p90 = backtest.quantile_loss(np.array([10.0]), np.array([14.0]), 0.9)[0]
    assert over_p10 > over_p90
    # Median loss is symmetric and equals 0.5 * |error|.
    assert backtest.quantile_loss(y, np.array([8.0]), 0.5)[0] == 1.0


def test_summarize_computes_mae_and_rmse():
    acc = backtest.Accumulator()
    # Two point predictions with errors 2 and 4 -> MAE 3, RMSE sqrt(10).
    acc.add("m", "CPH", 1, y=10.0, row=pd.Series({"mean": 8.0}))
    acc.add("m", "CPH", 1, y=10.0, row=pd.Series({"mean": 6.0}))
    summary = backtest.summarize(acc.frame())
    row = summary[(summary["model"] == "m") & (summary["horizon_h"] == 1)].iloc[0]
    assert row["MAE"] == 3.0
    assert abs(row["RMSE"] - np.sqrt(10.0)) < 1e-9
    assert np.isnan(row["WQL"])  # no quantiles supplied


def test_summarize_reports_wql_for_probabilistic_rows():
    acc = backtest.Accumulator()
    acc.add("p", "CPH", 1, y=10.0,
            row=pd.Series({"mean": 9.0, "q10": 7.0, "q50": 9.0, "q90": 12.0}))
    summary = backtest.summarize(acc.frame())
    row = summary[summary["horizon_h"] == 1].iloc[0]
    assert not np.isnan(row["WQL"])
    assert row["WQL"] >= 0.0


def test_lookup_actuals_respects_tolerance():
    idx = pd.date_range("2026-06-01", periods=5, freq="5min", tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="queue")
    # Exact hit and a far-away target (beyond tolerance -> NaN).
    targets = [idx[2], idx[-1] + pd.Timedelta(hours=3)]
    out = backtest.lookup_actuals(series, targets, tolerance_minutes=5)
    assert out.iloc[0] == 3.0
    assert np.isnan(out.iloc[1])


if __name__ == "__main__":
    import sys

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
