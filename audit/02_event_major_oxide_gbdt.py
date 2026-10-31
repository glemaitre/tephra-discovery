# %% [markdown]
# # Audit — 02_event_major_oxide_gbdt: do the major oxides separate the 30 eruptions?
#
# Read-only review of the stored report: checks and metrics for this
# experiment. The learner predicts the eruption from the 11 major oxides
# the table itself normalizes, and is compared with the dummy baseline.

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
REPORT_ID = "01a19295-7e42-734a-808d-9ba5c19da885"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Persisted report
#
# local workspace: [reports/](../reports/) · id: 01a19295-7e42-734a-808d-9ba5c19da885

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
# Headline scores for this experiment. The Brier score and the per-class
# ROC AUC are the locked comparison against the dummy baseline
# (Brier -1.849821, accuracy 0.075089).

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
