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


def test_wql_uses_the_common_quantiles_so_models_stay_comparable():
    """A 21-level model and a 3-level model must yield the same WQL quantity.

    Chronos-2 publishes 21 levels, XGBoost 3. WQL averages the per-level losses,
    so averaging over whatever each model happened to emit produces two different
    statistics printed in one column -- and the model with the wider, easier tail
    coverage looks better for no modelling reason. The headline WQL is therefore
    pinned to COMMON_QUANTILES, with the wide average reported separately.
    """
    narrow = pd.Series({"mean": 9.0, "q10": 7.0, "q50": 9.0, "q90": 12.0})
    wide = narrow.copy()
    # Extra levels, deliberately near-perfect, to make a wide average look great.
    for name in ("q1", "q5", "q95", "q99"):
        wide[name] = 10.0

    acc = backtest.Accumulator()
    acc.add("narrow", "CPH", 1, y=10.0, row=narrow)
    acc.add("wide", "CPH", 1, y=10.0, row=wide)
    frame = acc.frame()

    by_model = frame.set_index("model")
    assert by_model.loc["narrow", "q_loss"] == by_model.loc["wide", "q_loss"], (
        "WQL must ignore the extra levels the wide model published"
    )
    assert by_model.loc["wide", "q_loss_wide"] < by_model.loc["narrow", "q_loss_wide"]
    assert by_model.loc["narrow", "n_quantiles"] == 3
    assert by_model.loc["wide", "n_quantiles"] == 7


def test_all_rollup_restricted_to_horizons_every_model_reached():
    """Chronos stops at 8h; the baselines answer at 168h.

    Rolling every model up over its own horizon set would credit the short-reach
    model for skipping the hardest horizons, which reads as a large accuracy win.
    """
    acc = backtest.Accumulator()
    # Both models are perfect at 1h. Only the baseline answers at 168h, badly.
    acc.add("chronos2", "CPH", 1, y=10.0, row=pd.Series({"mean": 10.0}))
    acc.add("seasonal_naive", "CPH", 1, y=10.0, row=pd.Series({"mean": 10.0}))
    acc.add("seasonal_naive", "CPH", 168, y=10.0, row=pd.Series({"mean": 0.0}))

    summary = backtest.summarize(acc.frame())
    overall = summary[summary["horizon_h"] == "ALL"].set_index("model")

    # Without the restriction the baseline's ALL MAE would be 5.0 (mean of 0 and
    # 10) and Chronos would appear to win a comparison it never entered.
    assert overall.loc["seasonal_naive", "MAE"] == 0.0
    assert overall.loc["chronos2", "MAE"] == 0.0
    assert overall.loc["seasonal_naive", "n"] == 1
    assert overall.iloc[0]["all_horizons"] == "1"


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
