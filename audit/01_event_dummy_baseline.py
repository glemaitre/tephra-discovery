# %% [markdown]
# # Audit — 01_event_dummy_baseline: does the dummy baseline run end to end?
#
# Read-only review of the stored report: checks and metrics for this
# experiment. The dummy always predicts the most frequent eruption and
# is the reference the later learners have to beat.

# %%
import skore

from winter_school import PROJECT_ROOT

# %% [markdown]
# ## Open the project

# %%
project = skore.Project(
    name="volcanic-geochemistry",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)
project

# %% [markdown]
# ## List the available reports

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report

# %%
REPORT_ID = "01a1927f-a51d-76f4-814e-5a4eb30218d0"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Persisted report
#
# local workspace: [reports/](../reports/) · id: 01a1927f-a51d-76f4-814e-5a4eb30218d0

# %% [markdown]
# ## Checks summary
#
# Automated checks for this report: issues and tips that affect how
# to read the metrics below.

# %%
checks = report.checks.summarize()
checks

# %% [markdown]
# ## Metrics summary
#
# Headline scores for this experiment. The dummy predicts one class for
# every row, so the Brier score and the ROC AUC are the reference
# values, not evidence that the chemistry separates the 30 eruptions.

# %%
metrics = report.metrics.summarize().frame(verbose_name=True, flat_index=False)
# Per-class rows only: the locked comparison excludes macro and micro
# averages across classes.
if "Average" in metrics.index.names:
    metrics = metrics[metrics.index.get_level_values("Average") == ""]
metrics

# %% [markdown]
# ## Core audit complete
#
# Checks and metrics above are the first pass. Further views stay
# read-only on this same report.

# %% [markdown]
# ## ROC curve per event
#
# One curve per event, read-only on the same report. The dummy predicts
# class priors for every row, so each curve collapses onto the diagonal.

# %%
roc = report.metrics.roc()
roc
