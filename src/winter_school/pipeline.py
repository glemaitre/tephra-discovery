"""Learner declaration for the baseline experiment.

Owns: the function that builds and returns the unfit learner. The
graph roots on a `data_dir` source, selects the chemistry features,
and fits a dummy that always predicts the most frequent eruption. The
comparison metric is the multiclass Brier score, which skore reports
only for binary classification, so it is attached with
`with_scoring`. Fitting, evaluation, and persistence happen elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import skrub
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, make_scorer

from winter_school.data import TARGET_COL, load_raw
from winter_school.features import select_features, select_major_oxides

# Probability columns follow the estimator's sorted class order, which
# is also the order `brier_score_loss` infers from `y_true`.
brier_scorer = make_scorer(
    brier_score_loss,
    response_method="predict_proba",
    greater_is_better=False,
)


def build_learner(data_dir_preview: str | Path | None = None):
    """Return the unfit learner for the experiment scripts to consume.

    Parameters
    ----------
    data_dir_preview : str or Path or None, optional
        Preview value for the source-bound `skrub.var("data_dir", ...)`
        root. Pass an absolute path while iterating interactively so
        `learner.skb.preview()` works. Leave as `None` for fit runs —
        the env dict passed to `skore.evaluate` supplies the binding.

    Returns
    -------
    skrub.SkrubLearner
        The unfit learner.
    """
    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )
    data = data_dir.skb.apply_func(load_raw)
    X = data.skb.apply_func(select_features).skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()
    predictions = X.skb.apply(
        DummyClassifier(strategy="most_frequent", random_state=0), y=y
    )
    predictions = predictions.skb.with_scoring(brier_scorer, name="brier_score")
    return predictions.skb.make_learner()


def build_oxide_learner(data_dir_preview: str | Path | None = None):
    """Return the unfit major-oxide learner.

    The first learner that uses the chemistry: a histogram gradient-boosted
    classifier over the 11 major oxides, with no class weighting (the
    comparison is a proper scoring rule, so weighting would distort the
    predicted class frequencies) and no imputation step (missing oxides
    stay NaN and the learner bins them natively).

    Parameters
    ----------
    data_dir_preview : str or Path or None, optional
        Preview value for the source-bound `skrub.var("data_dir", ...)`
        root. Pass an absolute path while iterating interactively so
        `learner.skb.preview()` works. Leave as `None` for fit runs.

    Returns
    -------
    skrub.SkrubLearner
        The unfit learner.
    """
    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )
    data = data_dir.skb.apply_func(load_raw)
    X = data.skb.apply_func(select_major_oxides).skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()
    predictions = X.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)
    predictions = predictions.skb.with_scoring(brier_scorer, name="brier_score")
    return predictions.skb.make_learner()
