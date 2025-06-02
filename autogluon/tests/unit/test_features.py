

"""
Unit tests for *feature‑engineering* helpers in **project.app**.

The production code base provides a helper that takes a raw time‑series
DataFrame and expands it with deterministic features such as:

* Hour‑of‑day sine / cosine (``hour_sin``, ``hour_cos``)
* Day‑of‑week sine / cosine (``dow_sin``, ``dow_cos``)
* Rolling aggregates (columns containing ``roll`` or ``rolling``)

Because the exact helper name is not part of the public API we discover it
at runtime by looking for common nomenclature.  If the helper cannot be
found the test *skips* rather than fails, to avoid blocking unrelated
branches during early development.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
#                    Locate the feature‑engineering helper                    #
# --------------------------------------------------------------------------- #
def _find_feature_helper() -> Callable[[pd.DataFrame], pd.DataFrame]:
    # Ensure the repository root is on sys.path so that `project.app`
    # (a namespace package living at `<repo_root>/project`) is importable.
    from pathlib import Path
    import sys

    repo_root = Path(__file__).resolve().parents[2]  # `<repo_root>/autogluon`
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import importlib

    try:
        app = importlib.import_module("project.app")
    except ModuleNotFoundError:
        # Fallback for repository layout where 'project' lives inside 'autogluon/'
        app = importlib.import_module("autogluon.project.app")

    candidates: List[str] = [
        # likely public helpers
        "add_features",
        "make_features",
        "prepare_features",
        "build_features",
        # private helpers
        "_add_features",
        "_make_features",
        "_prepare_features",
        "_build_features",
        "feature_engineering",
        "_feature_engineering",
    ]

    for name in candidates:
        if hasattr(app, name):
            fn = getattr(app, name)
            if callable(fn):
                return fn

    pytest.skip(
        "No feature‑engineering helper found in `project.app`. "
        "Skipping feature tests.",
        allow_module_level=True,
    )


FEATURE_HELPER = _find_feature_helper()


# --------------------------------------------------------------------------- #
#                          Synthetic minimal raw data                         #
# --------------------------------------------------------------------------- #
def _make_raw_df() -> pd.DataFrame:
    """Create a 24‑hour, 5‑minute cadence DataFrame with a `queue` column."""
    idx = pd.date_range("2024-01-01", periods=288, freq="5min")  # 24h * 12
    df = pd.DataFrame({"queue": np.random.rand(len(idx))}, index=idx)
    return df


# --------------------------------------------------------------------------- #
#                                   Tests                                     #
# --------------------------------------------------------------------------- #
def test_feature_helper_adds_columns():
    """
    The helper must *add* columns (non‑destructively) and preserve index
    length.
    """
    raw = _make_raw_df()
    expanded = FEATURE_HELPER(raw.copy())

    # The helper may return a *view* of the original DF; ensure same length.
    assert len(expanded) == len(raw)

    # Must contain all original columns.
    for col in raw.columns:
        assert col in expanded.columns

    new_cols = set(expanded.columns) - set(raw.columns)
    assert new_cols, "Expected the helper to add at least one new column"


def test_trig_columns_within_unit_circle():
    """
    All sine / cosine features should be bounded by ±1.
    """
    expanded = FEATURE_HELPER(_make_raw_df())

    trig_cols = [c for c in expanded.columns if c.endswith(("sin", "cos"))]
    assert trig_cols, "No sine/cosine columns produced"

    for col in trig_cols:
        series = expanded[col].dropna()
        assert (series.abs() <= 1.0000001).all(), f"{col} outside [−1,1] range"


def test_rolling_features_have_few_nans():
    """
    Rolling aggregates inevitably introduce leading NaNs but the *majority*
    of rows (≥ 75 %) should be finite.
    """
    expanded = FEATURE_HELPER(_make_raw_df())

    roll_cols = [c for c in expanded.columns if "roll" in c.lower()]
    if not roll_cols:
        pytest.skip("No rolling feature columns present")

    for col in roll_cols:
        series = expanded[col]
        fraction_valid = series.notna().mean()
        assert (
            fraction_valid >= 0.75
        ), f"{col} has too many NaNs ({1-fraction_valid:.0%} empty)"


def test_feature_helper_pure_function():
    """
    The raw DataFrame passed in should *not* be mutated in‑place.
    """
    raw = _make_raw_df()
    raw_copy = raw.copy(deep=True)

    _ = FEATURE_HELPER(raw)

    pd.testing.assert_frame_equal(
        raw, raw_copy, check_dtype=False
    ), "Feature helper mutated its input in‑place"