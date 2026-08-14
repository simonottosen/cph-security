"""Serving app for the global XGBoost quantile predictor.

This process only *serves* predictions. Training happens out-of-process in
``train.py`` (run as a cron / k8s CronJob); this app loads the resulting
``model.joblib`` bundle from ``MODELS_DIR`` and hot-reloads it whenever the file
changes on disk, so every replica reads the same artifact and a fresh model is
picked up without a restart.

Contract (unchanged, plus new quantile fields):
    GET /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM   (timestamp in UTC)
    -> {"predicted_queue_length_minutes": <int p50>,
        "p10": .., "p50": .., "p90": ..,
        "airport": "ARN", "timestamp": "...Z"}
"""
from __future__ import annotations

import logging
import os
import threading
import time

import pandas as pd
from flask import Flask, jsonify, request

import model as model_mod
from features import VALID_AIRPORTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)

# ------------------------------------------------------------------------------
# Bundle loading with mtime-based hot-reload
# ------------------------------------------------------------------------------
_bundle = None
_bundle_mtime = None
_bundle_lock = threading.Lock()


def _current_mtime() -> float | None:
    try:
        return os.path.getmtime(model_mod.model_path(MODELS_DIR))
    except OSError:
        return None


def get_bundle():
    """Return the active bundle, reloading if the artifact changed on disk.

    Falls back to a ``ConstantBundle`` until a real model has been trained, so
    ``/predict`` always answers.
    """
    global _bundle, _bundle_mtime
    mtime = _current_mtime()
    if _bundle is not None and mtime == _bundle_mtime:
        return _bundle

    with _bundle_lock:
        mtime = _current_mtime()
        if _bundle is not None and mtime == _bundle_mtime:
            return _bundle
        loaded = model_mod.load_bundle(MODELS_DIR)
        if loaded is None:
            if _bundle is None:
                logger.warning("No model at %s; serving constant fallback.",
                               model_mod.model_path(MODELS_DIR))
                _bundle = model_mod.ConstantBundle()
        else:
            _bundle = loaded
            _bundle_mtime = mtime
            logger.info("Loaded model bundle (trained_at=%s, metrics=%s)",
                        getattr(loaded, "trained_at", "?"),
                        getattr(loaded, "metrics", {}))
        return _bundle


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/predict")
def make_prediction():
    start_time = time.time()

    input_date_str = request.args.get("timestamp")
    airport_code = request.args.get("airport")

    if not input_date_str and not airport_code:
        return jsonify({
            "error": 'Missing "airport" and "timestamp". '
                     "e.g., /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM"
        }), 400
    if not input_date_str:
        return jsonify({
            "error": 'Missing "timestamp". e.g., /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400
    if not airport_code:
        return jsonify({
            "error": 'Missing "airport". e.g., /predict?airport=ARN&timestamp=YYYY-MM-DDTHH:MM'
        }), 400

    airport_code = airport_code.upper()
    if airport_code not in VALID_AIRPORTS:
        return jsonify({
            "error": f'Invalid airport code "{airport_code}". '
                     f'Valid codes: {",".join(VALID_AIRPORTS)}.'
        }), 400

    try:
        input_date = pd.to_datetime(input_date_str, utc=True)
    except (ValueError, TypeError):
        return jsonify({"error": 'Invalid "timestamp". Expected YYYY-MM-DDTHH:MM (UTC)'}), 400

    bundle = get_bundle()
    frame = bundle.predict_frame(airport_code, [input_date])
    row = frame.iloc[0]

    cols = list(frame.columns)
    p50_col = "q50" if "q50" in cols else cols[len(cols) // 2]
    p50 = float(row[p50_col])

    response = {
        "predicted_queue_length_minutes": int(round(p50)),
        "airport": airport_code,
        "timestamp": input_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Expose each quantile as p10/p50/p90 (mapped from q10/q50/q90).
    for col in cols:
        response["p" + col[1:]] = round(float(row[col]), 1)

    logger.info("Prediction for %s completed in %.3fs", airport_code, time.time() - start_time)
    return jsonify(response)


@app.route("/health")
def health():
    bundle = get_bundle()
    return jsonify({
        "status": "ok",
        "model": type(bundle).__name__,
        "trained_at": getattr(bundle, "trained_at", None),
        "quantiles": getattr(bundle, "quantile_columns", []),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
