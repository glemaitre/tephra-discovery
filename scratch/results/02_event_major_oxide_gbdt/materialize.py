"""Agent-only materialize. Copy of the evaluate-ml-pipeline template.

Do not commit. This is the only evaluation run. Do not execute
``experiments/02_event_major_oxide_gbdt.py`` and do not
``notebook convert`` it. The ``CALL`` block mirrors that file's fit /
``skore.evaluate`` / ``report.checks.summarize()`` / ``project.put``,
same arguments and order.
"""

import inspect
import shutil

import skore
from sklearn.utils import estimator_html_repr

from winter_school import PROJECT_ROOT
from winter_school.pipeline import build_oxide_learner

STEM = "02_event_major_oxide_gbdt"

project = skore.Project(
    name="volcanic-geochemistry",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)

test_dir = PROJECT_ROOT / "scratch" / "results" / STEM / "test_table"
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
# </CALL>

results = PROJECT_ROOT / "scratch" / "results" / STEM
results.mkdir(parents=True, exist_ok=True)
checks = report.checks.summarize()
metrics = report.metrics.summarize().frame(verbose_name=True, flat_index=False)
# Keep the per-class rows only: drop the macro and micro averages across
# classes, which skore adds for the multiclass task.
if "Average" in metrics.index.names:
    metrics = metrics[metrics.index.get_level_values("Average") == ""]
if hasattr(metrics, "to_frame"):
    metrics = metrics.to_frame()
metrics = metrics.drop(
    columns=[column for column in metrics.columns if column.endswith("_avg")],
    errors="ignore",
)
(results / "report.html").write_text(report._repr_html_(), encoding="utf-8")
(results / "report.txt").write_text(repr(report) + "\n", encoding="utf-8")
(results / "checks.html").write_text(checks._repr_html_(), encoding="utf-8")
(results / "metrics.html").write_text(metrics._repr_html_(), encoding="utf-8")
summary = project.summarize().frame()
report_id = summary.loc[summary["key"] == STEM, "report_id"].iloc[-1]
(results / "id.txt").write_text(f"{report_id}\n", encoding="utf-8")

learner = (
    report.estimator_
    if hasattr(report, "estimator_")
    else report.reports_[0].estimator_
)
data_op = getattr(getattr(learner, "data_op", None), "skb", None)
method = getattr(data_op, "report", None)
if callable(method) and "eval" in inspect.signature(method).parameters:
    learner.report(
        eval=False,
        open=False,
        overwrite=True,
        output_dir=results / "pipeline",
    )
else:
    render = getattr(learner, "_repr_html_", None)
    (results / "pipeline.html").write_text(
        render() if callable(render) else estimator_html_repr(learner),
        encoding="utf-8",
    )
