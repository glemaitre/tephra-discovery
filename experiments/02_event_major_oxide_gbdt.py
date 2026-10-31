# %% [markdown]
# # 02 — major-oxide gradient boosting for eruption attribution
#
# The first learner that uses the chemistry. It predicts `Event` (30
# classes) from the 11 major oxides the table itself normalizes, and is
# compared against the most-frequent-event dummy on the locked metric:
# the multiclass Brier score plus per-class ROC AUC, with no average
# across classes.
#
# Two deliberate omissions. No class weighting, because the goal is
# probabilities and a proper scoring rule — weighting would distort the
# predicted class frequencies and worsen the Brier score, so the 8.5:1
# imbalance is read through the per-class rows instead. No imputation,
# because the learner bins missing oxides natively, and 1282 of the 6220
# training rows carry no oxides at all.
#
# The split is the shipped one: the training table is loaded here, and
# the test table is bound later at evaluation time.

# %%
from winter_school import PROJECT_ROOT
from winter_school.pipeline import build_oxide_learner

STEM = "02_event_major_oxide_gbdt"
results = PROJECT_ROOT / "scratch" / "results" / STEM
results.mkdir(parents=True, exist_ok=True)

learner = build_oxide_learner(data_dir_preview=PROJECT_ROOT / "data")
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
# The classifier is fitted on the training table and scored on the
# shipped test table with `splitter="prefit"`, so skore draws no new
# split and the training table is never passed to the evaluation. The
# comparison is the multiclass Brier score attached to the graph, next
# to the per-class ROC AUC skore reports. The report is stored in the
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

learner = build_oxide_learner()
fitted = learner.fit({"data_dir": str(PROJECT_ROOT / "data")})
report = skore.evaluate(
    fitted,
    data={"data_dir": str(test_dir)},
    splitter="prefit",
)
report.checks.summarize()
project.put(STEM, report)
report
