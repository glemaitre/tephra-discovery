# 02_event_major_oxide_gbdt

## Question / hypothesis

Can a gradient-boosted classifier on the 11 major oxides beat the
most-frequent-event dummy on the locked comparison — the multiclass Brier
score and per-class ROC AUC for `Event`?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - the recorded EDA report — major oxides test at p ≈ 0 against `Event`
    (`SiO2`, `K2O`, `MgO`, `FeOT`, `Al2O3`), between-source variance is 15–26×
    the within-sample variance over 222 replicated samples, and the identity
    check shows `Total` and every `*_normalized` column are arithmetic
    duplicates of the oxides
  - the locked Modeling decisions block — goal `probabilities`, comparison
    "Brier score over rows and classes plus per-class ROC AUC, with no macro or
    micro average across classes", `prefit` split, baseline `dummy` /
    "always predict PCC2011"
  - experiment 01 — the dummy reference sits at chance (Brier -1.849821,
    per-class ROC AUC 0.500, accuracy 0.075089)
- **Why this matters:** the baseline proved the data path and fixed the
  reference, but it ignores every feature. This is the first experiment that
  asks whether the chemistry separates the 30 eruptions at all.

## Method

- **Files touched:** `src/winter_school/features.py`,
  `src/winter_school/pipeline.py`, `experiments/02_event_major_oxide_gbdt.py`,
  `tests/smoke/test_02_event_major_oxide_gbdt.py`
- **Change versus baseline (or previous experiment):** the predictor changes
  from `DummyClassifier(strategy="most_frequent")` to
  `HistGradientBoostingClassifier`, and the feature block narrows from all 45
  chemistry columns to the 11 major oxides the table itself normalizes
  (`SiO2`, `TiO2`, `Al2O3`, `FeOT`, `MnO`, `MgO`, `CaO`, `Na2O`, `K2O`,
  `P2O5`, `Cl`). No class weighting is used, because the goal is
  probabilities and a proper scoring rule: weighting would distort the
  predicted class frequencies and worsen the Brier score, so the 8.5:1
  imbalance is read through the per-class rows instead. Missing oxides are
  left as NaN, which the learner handles natively, so no imputation step is
  introduced.
- **Cross-validation:** the scheme locked under Modeling decisions
  in the journal.
- **Out of scope for this experiment:** trace elements and ratios, isotope
  columns, a log-ratio transform of the compositional oxide block,
  hyperparameter tuning, post-hoc calibration of the emitted probabilities,
  class weighting, and a grouped-by-`SampleID` evaluation.
- **Pipeline:** <!-- results-embed: pipeline -->

## Risks / things that could invalidate the result

- **1282 of 6220 training rows carry none of the 11 oxides** — 1155 of them
  carry trace elements instead, and 1091 are LA-ICP-MS analyses. Those rows
  present an all-NaN feature vector, so the learner can only answer them from
  the prior. All 30 events still appear among the 4938 oxide-carrying rows, so
  the target stays learnable, but a fifth of the table is invisible to this
  model.
- **The shipped split keeps all 30 events in training**, so the score measures
  new analyses of known eruptions. A good result is consistent with the model
  memorizing event-level composition rather than generalizing to an unseen
  eruption.
- **Dropping the trace block gives up the discriminating elements.** The
  closest source pairs in major-oxide space are Michinmahuida/Quetrupillán
  (1.03) and Chaitén/Reclus (1.06), and the literature reports trace elements
  are what resolve overlapping major-element signatures.
- **The 11 oxides are compositional.** They sum to `Total` within 1.4e-14, so
  their variation is not independent and the block is effectively
  lower-dimensional than 11 columns.
- **Rare events stay hard.** `Quet1` has 67 rows against 567 for `PCC2011`,
  and without class weighting a rare event may keep near-zero predicted
  probability; the per-class rows are what will show it.

## Status

- **State:** approved
- **Approved by user on:** 2026-10-31
- **Headline result:** Brier score -0.360931 (accuracy 0.723480, per-class ROC AUC 0.872–1.000)
- **Persisted report:** local workspace: [reports/](../reports/) · id: 01a19295-7e42-734a-808d-9ba5c19da885<br>absolute path: /Users/glemaitre/Documents/scratch/winter-school/reports/
- **Audit findings:** 0 issue(s), 1 tip(s) — SKD016 (tip); accuracy 0.723480
- **Implication for next iteration:** the major oxides alone separate the 30 events by a wide margin, so the next questions are whether the untuned defaults are what caps the score and whether the same result holds when the trace block is added or when the evaluation groups by `SampleID`.

## Results

### Report overview

The major-oxide classifier beats the dummy reference on the locked comparison: multiclass Brier score -0.361 against -1.850, accuracy 0.723 against 0.075, and log loss 1.140. The per-class ROC AUC runs from 0.872 for the hardest event to 1.000 for the easiest, so almost every event ranks its own samples above the rest even though all 30 events appear in both the training and test tables. The score comes from the shipped training and test tables: the model is fitted on the 6220 training rows and scored on the 839 test rows, which share no sample identifier.

### Checks

The automated checks raised 0 issues and 1 tip: the estimator is left at its default settings and skore suggests tuning `learning_rate` and `max_leaf_nodes` for `HistGradientBoostingClassifier`. Fifteen checks came back not applicable; for eight of them the stated reason is that train data is unavailable, because a `prefit` report carries only the test bindings. That is the same limitation that left the dummy's train-side checks unrun, so the summary still cannot separate "no problem" from "never checked" for overfitting, underfitting, correlated features and the baseline comparisons.

### Metrics

## Notebooks

### Evaluation notebook

### Audit notebook
