"""API-contract tests for the serving app.

These are hermetic: a tiny model is trained from synthetic history into a temp
``MODELS_DIR`` before the app is imported, so no network or on-disk artifact is
needed. They lock in the ``/predict`` schema (legacy ``predicted_queue_length_
minutes`` plus new ``p10/p50/p90``) and the input-validation behaviour.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make the serving modules importable when run standalone (mirrors conftest).
_PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"
)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def _synthetic_history(base, n=4000, seed=0):
    idx = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    vals = base + 3.0 * np.sin(np.arange(n) / 40.0) + rng.normal(0, 0.5, n)
    return pd.Series(np.maximum(0.0, vals), index=idx, name="queue")


def _build_model(models_dir):
    import features as f
    import model as m

    histories = {"CPH": _synthetic_history(6.0, seed=1),
                 "ARN": _synthetic_history(9.0, seed=2)}
    profiles = {ap: f.build_profiles(s, ap) for ap, s in histories.items()}
    x, y = f.build_training_frame(histories, profiles)
    booster, qa, _ = m.train_booster(
        x, y, num_boost_round=40, early_stopping_rounds=None
    )
    bundle = m.ModelBundle(
        booster=booster,
        feature_names=f.FEATURE_COLUMNS + ["airport"],
        quantile_alpha=qa,
        profiles=profiles,
        trained_at="test",
        metrics={},
        best_iteration=None,
    )
    m.save_bundle(bundle, models_dir)


# Set up the model + env BEFORE importing the app (it reads MODELS_DIR at import).
_TMP_MODELS = tempfile.mkdtemp(prefix="ml_api_test_")
os.environ["MODELS_DIR"] = _TMP_MODELS
_build_model(_TMP_MODELS)

import app as app_mod  # noqa: E402

client = app_mod.app.test_client()


def test_predict_returns_legacy_and_quantile_fields():
    r = client.get("/predict?airport=CPH&timestamp=2026-07-03T07:30")
    assert r.status_code == 200
    body = r.get_json()
    assert "predicted_queue_length_minutes" in body
    assert isinstance(body["predicted_queue_length_minutes"], int)
    for key in ("p10", "p50", "p90"):
        assert key in body
    assert body["airport"] == "CPH"
    assert body["timestamp"].endswith("Z")
    # Legacy field equals rounded p50.
    assert body["predicted_queue_length_minutes"] == round(body["p50"])


def test_predict_quantiles_monotonic_and_nonnegative():
    r = client.get("/predict?airport=ARN&timestamp=2026-07-05T17:00")
    body = r.get_json()
    assert 0.0 <= body["p10"] <= body["p50"] <= body["p90"]


def test_predict_lowercase_airport_is_accepted():
    r = client.get("/predict?airport=cph&timestamp=2026-07-03T07:30")
    assert r.status_code == 200
    assert r.get_json()["airport"] == "CPH"


def test_predict_missing_params_return_400():
    assert client.get("/predict").status_code == 400
    assert client.get("/predict?airport=CPH").status_code == 400
    assert client.get("/predict?timestamp=2026-07-03T07:30").status_code == 400


def test_predict_invalid_airport_returns_400():
    r = client.get("/predict?airport=ZZZ&timestamp=2026-07-03T07:30")
    assert r.status_code == 400


def test_predict_invalid_timestamp_returns_400():
    r = client.get("/predict?airport=CPH&timestamp=not-a-date")
    assert r.status_code == 400


def test_health_reports_loaded_model():
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["model"] == "ModelBundle"
    assert body["quantiles"] == ["q10", "q50", "q90"]


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
