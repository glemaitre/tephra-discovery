"""Smoke test for `experiments/02_event_major_oxide_gbdt.py`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winter_school import PROJECT_ROOT
from winter_school.pipeline import build_oxide_learner

DATA_DIR = PROJECT_ROOT / "data"
SMOKE_ROWS = 40


@pytest.fixture
def train_predict_envs(tmp_path):
    """Build the fit and predict environments from the real table.

    Diagnostic by construction: the predict environment holds only the
    rows we want predictions for, with no extra history.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.

    Returns
    -------
    tuple
        ``(train_env, predict_env, n_predict_grid_rows, y_true)``.
    """
    frame = pd.read_csv(DATA_DIR / "train.csv")
    predict_frame = frame.tail(SMOKE_ROWS).reset_index(drop=True)
    train_frame = frame.iloc[:-SMOKE_ROWS].reset_index(drop=True)

    train_dir = tmp_path / "train"
    predict_dir = tmp_path / "predict"
    train_dir.mkdir()
    predict_dir.mkdir()
    train_frame.to_csv(train_dir / "train.csv", index=False)
    predict_frame.to_csv(predict_dir / "train.csv", index=False)

    train_env = {"data_dir": str(train_dir)}
    predict_env = {"data_dir": str(predict_dir)}
    y_true = predict_frame["Event"].to_numpy()
    return train_env, predict_env, len(predict_frame), y_true


def test_02_event_major_oxide_gbdt(train_predict_envs):
    """Predict-time replay returns one well-formed row per predict-grid row."""
    train_env, predict_env, n_predict_grid_rows, _y_true = train_predict_envs

    learner = build_oxide_learner()
    learner.fit(train_env)
    predictions = learner.predict(predict_env)

    assert len(predictions) == n_predict_grid_rows, (
        f"got {len(predictions)} predictions for {n_predict_grid_rows} "
        f"predict-grid rows."
    )
    width = predictions.shape[1] if getattr(predictions, "ndim", 1) == 2 else 1
    assert width == 1, f"got {width} outputs, expected 1"

    # Soft bound in place of a metric-vs-CV gap: the design note is still
    # `planned`, so there is no headline number to compare against yet. What
    # the bound still catches is NaN-poisoned probabilities — the failure
    # mode where the row count is right and the values are garbage.
    proba = learner.predict_proba(predict_env)
    scores = proba.to_numpy() if hasattr(proba, "to_numpy") else np.asarray(proba)
    assert scores.shape[0] == n_predict_grid_rows, (
        f"got {scores.shape[0]} probability rows for "
        f"{n_predict_grid_rows} predict-grid rows."
    )
    assert np.isfinite(scores).all(), "predicted probabilities are not finite"
    assert ((scores >= 0) & (scores <= 1)).all(), "probabilities outside [0, 1]"
    assert np.allclose(scores.sum(axis=1), 1.0), "probability rows do not sum to 1"
