"""
Train and compare Chronos-Bolt (Base) zero-shot vs fine-tuned models per airport.

Changes vs previous version:
- Removes Flask API and scheduler; this is a simple script runnable via `python fine_tune.py`.
- Loads local data from `all.json` instead of calling a remote API.
- Uses AutoGluon TimeSeries Chronos model with *two* configurations:
  1) Zero-shot (inference-only)
  2) Fine-tuned (lightweight fine-tuning)
- Prints a concise leaderboard comparison per airport and prepares predictions for both models.

Note: `model_path="bolt_base"` corresponds to the Chronos-Bolt Base model.
"""

import os
import time
import logging
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


# ------------------------
# Setup & constants
# ------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/ts_predictor"
PREDICTION_LENGTH = 96  # 96 x 5min = 8 hours ahead
FREQ = "5min"

# Valid airports (restrict training to these if present in the data)
VALID_AIRPORTS = [
    "CPH"
]


# ------------------------
# Utilities
# ------------------------

def _prepare_ts(df_raw: pd.DataFrame, code: str) -> TimeSeriesDataFrame:
    """Prepare a univariate time series for a single airport.

    Expects df_raw to have columns: ['airport', 'timestamp', 'queue']
    Returns a TimeSeriesDataFrame with 5-min frequency.
    """
    df_code = df_raw[df_raw["airport"] == code].copy()
    if df_code.empty:
        raise ValueError(f"No data found for airport {code}")

    # Parse / clean
    df_code["timestamp"] = pd.to_datetime(df_code["timestamp"], errors="coerce")
    df_code["queue"] = pd.to_numeric(df_code["queue"], errors="coerce")
    df_code = df_code.dropna(subset=["timestamp", "queue"]).set_index("timestamp")

    # Aggregate duplicate timestamps then resample to uniform 5-min grid
    df_code = df_code[["queue"]].groupby(level=0).mean()
    df_resampled = df_code.resample(FREQ, origin="start_day").ffill()

    # Build TimeSeriesDataFrame (target column only)
    df_resampled.index.name = "timestamp"
    df_resampled = df_resampled.reset_index()
    df_resampled["item_id"] = code
    df_resampled["timestamp"] = df_resampled["timestamp"].dt.tz_localize(None)

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df_resampled[["item_id", "timestamp", "queue"]],
        id_column="item_id",
        timestamp_column="timestamp",
    )
    return ts_df


def _rename_quantile_cols(df: pd.DataFrame) -> list[dict]:
    """Standardize quantile column names to qXX and return list-of-dicts records.
    Expects `df` to be a pandas DataFrame with a DatetimeIndex named 'timestamp'.
    """
    out = df.reset_index()
    # Normalize timestamp to string
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        out.rename(columns={"index": "timestamp"}, inplace=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Map numeric/float columns like 0.1 -> q10
    rename_map = {}
    for c in list(out.columns):
        if isinstance(c, float):
            rename_map[c] = f"q{int(round(c * 100)):02d}"
        elif isinstance(c, str):
            try:
                f = float(c)
                rename_map[c] = f"q{int(round(f * 100)):02d}"
            except ValueError:
                pass
    out = out.rename(columns=rename_map)

    keep = ["timestamp", "mean"] + [c for c in out.columns if c.startswith("q")]
    keep = [c for c in keep if c in out.columns]
    return out[keep].to_dict(orient="records")


def train_and_compare_for_airport(code: str, df_raw: pd.DataFrame) -> tuple[dict, dict]:
    """Train Chronos-Bolt (Base) zero-shot and fine-tuned models for a single airport,
    return (metrics, predictions_by_model).
    """
    ts_df = _prepare_ts(df_raw, code)
    train_data, test_data = ts_df.train_test_split(prediction_length=PREDICTION_LENGTH)

    airport_path = os.path.join(MODEL_PATH, code)
    os.makedirs(airport_path, exist_ok=True)

    hyperparameters = {
        "Chronos": [
            {"model_path": "bolt_base", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_base", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    }

    start = time.time()
    predictor = TimeSeriesPredictor(
        path=airport_path,
        prediction_length=PREDICTION_LENGTH,
        target="queue",
        freq=FREQ,
        eval_metric="MAPE",
        verbosity=2,
        log_to_file=False,
    ).fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=600,  # seconds
        enable_ensemble=False,
    )
    train_seconds = time.time() - start

    # Identify model names
    model_names = predictor.model_names()
    zero_models = [m for m in model_names if "ZeroShot" in m]
    ft_models = [m for m in model_names if "FineTuned" in m]
    model_zero = zero_models[0] if zero_models else model_names[0]
    model_ft = ft_models[0] if ft_models else model_names[-1]

    # Leaderboard on the test split
    lb = predictor.leaderboard(test_data, display=False)

    # Forecast the next horizon from the latest observation using each model
    preds_zero = predictor.predict(ts_df, model=model_zero).loc[code]
    preds_ft = predictor.predict(ts_df, model=model_ft).loc[code]

    # Prepare artifacts directory
    out_dir = os.path.join(MODEL_PATH, code, "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    def to_records(df):
        df_reset = df.reset_index()
        # normalize timestamp -> ISO
        if "timestamp" in df_reset.columns:
            df_reset["timestamp"] = pd.to_datetime(df_reset["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        # rename quantile columns like 0.1 -> q10
        rename_map = {}
        for c in list(df_reset.columns):
            try:
                f = float(c)
                rename_map[c] = f"q{int(round(f*100)):02d}"
            except Exception:
                pass
        df_reset = df_reset.rename(columns=rename_map)
        keep = ["timestamp", "mean"] + [c for c in df_reset.columns if c.startswith("q")]
        return df_reset[keep].to_dict(orient="records")

    # Package metrics BEFORE saving to disk
    metrics = {
        "airport": code,
        "training_time_seconds": round(train_seconds, 2),
        "models_trained": model_names,
        "zero_shot_model_name": model_zero,
        "fine_tuned_model_name": model_ft,
        "leaderboard_test": lb.to_dict(orient="records"),
    }

    # Save artifacts
    lb.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)
    pd.Series(metrics).to_json(os.path.join(out_dir, "metrics.json"), orient="index")
    pd.DataFrame(to_records(preds_zero)).to_json(os.path.join(out_dir, "predictions_ZeroShot.json"), orient="records")
    pd.DataFrame(to_records(preds_ft)).to_json(os.path.join(out_dir, "predictions_FineTuned.json"), orient="records")

    # Build return payloads
    predictions = {
        "ZeroShot": _rename_quantile_cols(preds_zero),
        "FineTuned": _rename_quantile_cols(preds_ft),
    }

    # Release model resources
    del predictor

    return metrics, predictions

# ------------------------
# Entry point
# ------------------------

def main() -> None:
    # Load local data
    data_path = "/content/all.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected local file at {data_path}")

    df_raw = pd.read_json(data_path)
    required_cols = {"airport", "timestamp", "queue"}
    if not required_cols.issubset(df_raw.columns):
        raise ValueError("all.json must contain columns: 'airport', 'timestamp', 'queue'")

    # Focus on known airports present in the data
    airports_in_data = sorted(set(df_raw["airport"]).intersection(VALID_AIRPORTS))
    if not airports_in_data:
        raise ValueError("No valid airports found in all.json. Check 'airport' codes.")

    # Train & compare per airport
    for code in airports_in_data:
        logger.info(f"Training Chronos-Bolt (Base): zero-shot vs fine-tuned for {code} ...")
        metrics, predictions = train_and_compare_for_airport(code, df_raw)

        # Print a brief summary to stdout
        print(f"\n=== {code} ===")
        print(f"Trained models: {', '.join(metrics['models_trained'])}")
        print(f"Zero-shot model: {metrics['zero_shot_model_name']}")
        print(f"Fine-tuned model: {metrics['fine_tuned_model_name']}")

        # Summarize leaderboard (show common score columns if available)
        lb_df = pd.DataFrame(metrics["leaderboard_test"]) if metrics.get("leaderboard_test") else pd.DataFrame()
        if not lb_df.empty:
            cols = ["model"] + [c for c in lb_df.columns if "score" in c or c.upper() in {"MAE", "MAPE", "RMSE"}]
            cols = [c for c in cols if c in lb_df.columns]
            print(lb_df[cols].to_string(index=False))

        print(f"Prepared {PREDICTION_LENGTH} step-ahead forecasts at {FREQ} frequency for both models.\n")


if __name__ == "__main__":
    main()