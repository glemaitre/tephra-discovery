# 01_event_dummy_baseline

## Question / hypothesis

Can a dummy classifier that always predicts the most frequent eruption
(`PCC2011`) be loaded, fitted, predicted and smoke-tested end to end on the
volcanic geochemistry table, and what Brier score and ROC AUC does it set as
the reference for the 30-class `Event` target?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - the locked Modeling decisions block in the journal — baseline `dummy`,
    "always predict PCC2011", comparison "Brier score plus ROC AUC",
    deployment `groups`, generalize to `SampleID`, folds `predefined`
  - the recorded EDA report — 30 `Event` classes from 567 rows (PCC2011) down
    to 67 (Quet1); all 30 events appear in both the training and test tables,
    and the two share no `SampleID`
- **Why this matters:** it fixes the reference the later learners have to
  beat and proves the data path — loading, fit, predict, and the smoke
  assertion — before any real estimator is introduced.

## Method

- **Files touched:** `src/winter_school/data.py`,
  `src/winter_school/features.py`, `src/winter_school/pipeline.py`,
  `experiments/01_event_dummy_baseline.py`,
  `tests/smoke/test_01_event_dummy_baseline.py`
- **Change versus baseline (or previous experiment):** this is the baseline.
  The predictor is a dummy that predicts the most frequent `Event` label and
  emits no usable probabilities beyond the training class frequencies.
- **Cross-validation:** the scheme locked under Modeling decisions in the
  journal.
- **Out of scope for this experiment:** no real learner, no feature
  engineering beyond loading the chemistry table, no tuning, no imputation
  strategy.
- **Pipeline:** <!-- results-embed: pipeline -->

## Risks / things that could invalidate the result

- The dummy ignores the features, so its Brier score and ranking metric are
  the class-frequency reference; they say nothing about how separable the 30
  eruptions are.
- The test table holds all 30 events but the rarest has 67 rows, so a
  per-class reading of the reference is noisy for the tail classes.
- The trace-element columns cover about 20% of rows and `Sr87_Sr86` only 9
  rows, so the feature set actually fed to the pipeline is smaller than the
  45 chemistry columns the exploration counted.

## Status

- **State:** approved
- **Approved by user on:** 2026-10-31
- **Headline result:** Brier score -1.849821 (per-class ROC AUC 0.500)
- **Persisted report:** local workspace: [reports/](../reports/) · id: 01a1927f-a51d-76f4-814e-5a4eb30218d0<br>absolute path: /Users/glemaitre/Documents/scratch/winter-school/reports/
- **Audit findings:** 0 issues, 0 tips — automated checks surfaced no actionable finding; accuracy 0.075089
- **Implication for next iteration:** the reference sits at chance, so a real learner is the only way to learn whether the chemistry separates the 30 eruptions; the train-side checks cannot run at all until the report carries its training bindings.

## Results

### Report overview

The dummy baseline scores at chance on the 30-event target: accuracy 0.075, multiclass Brier score -1.850, log loss 33.34, and a per-class ROC AUC of 0.500 for every event. That is the expected shape for a predictor that ignores its features and always answers `PCC2011`, so the numbers are the reference a real learner has to beat rather than evidence about how separable the eruptions are. The split is the shipped training and test tables, and the two share no sample identifier.

### Checks

The automated checks raised 0 issues and 0 tips. Sixteen checks came back not applicable, and for eight of them the stated reason was that train data is unavailable, because a `prefit` report carries only the test bindings. The summary therefore cannot separate "no problem" from "never checked", which is why the train-side checks are recorded as an idea rather than as a clean bill of health.

### Metrics

### ROC curve per event

The ROC curves are read per event from the same loaded report, with no macro or micro average. Because the dummy emits class priors rather than a fitted score, every curve lies on the diagonal at AUC 0.500, which matches the per-class ROC AUC rows in the metrics table.

<!-- results-embed: roc -->

## Notebooks

### Evaluation notebook

### Audit notebook
