# %% [markdown]
# # 01 — dummy baseline for eruption attribution
#
# The reference for the `Event` target. The predictor is a dummy that
# always predicts the most frequent eruption (`PCC2011`), fitted on the
# 45 chemistry columns of the training table. It is not expected to add
# predictive value: it proves that loading, feature selection, fit,
# predict, scoring and the smoke assertion all run, and it fixes the
# Brier score and ROC AUC that a real learner has to beat.
#
# The split is the shipped one: the training table is loaded here, and
# the test table is bound later at evaluation time. The comparison
# metric is the multiclass Brier score, which skore reports only for
# binary classification, so it is attached to the graph with
# `with_scoring`.

# %%
from winter_school import PROJECT_ROOT
from winter_school.pipeline import build_learner

STEM = "01_event_dummy_baseline"
results = PROJECT_ROOT / "scratch" / "results" / STEM
results.mkdir(parents=True, exist_ok=True)

learner = build_learner(data_dir_preview=PROJECT_ROOT / "data")
learner.report(
    eval=False,
    open=False,
    overwrite=True,
    output_dir=results / "pipeline",
)
learner

# %% [markdown]
# ## Evaluation
#
# The dummy is fitted on the training table and scored on the shipped
# test table with `splitter="prefit"`, so skore draws no new split and
# the training table is never passed to the evaluation. The comparison
# is the multiclass Brier score attached to the graph, next to the ROC
# AUC skore reports for a multiclass target. The report is stored in the
# workspace under `reports/`.

# %%
import shutil

import skore

project = skore.Project(
    name="volcanic-geochemistry",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)

test_dir = results / "test_table"
test_dir.mkdir(parents=True, exist_ok=True)
shutil.copyfile(PROJECT_ROOT / "data" / "test.csv", test_dir / "train.csv")

learner = build_learner()
fitted = learner.fit({"data_dir": str(PROJECT_ROOT / "data")})
report = skore.evaluate(
    fitted,
    data={"data_dir": str(test_dir)},
    splitter="prefit",
)
report.checks.summarize()
project.put(STEM, report)
report
