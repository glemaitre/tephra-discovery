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
- **Headline result:** <fill in after run, or `n/a — abandoned: <reason>`>
- **Persisted report:** <backend locator after `project.put`, or `n/a — backend did not expose a locator`>
- **Audit findings:** n/a — audit not run
- **Implication for next iteration:** <fill in after run — what it suggests to try next. For abandonment: one line on what the abandonment teaches (e.g., "rules out monotonic-NN direction without paid GPU env")>

## Notebooks

### Evaluation notebook

### Audit notebook
