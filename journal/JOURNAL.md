# JOURNAL

## Status

| Variable | Value |
|---|---|
| Project / dataset | volcanic geochemistry — 30-event source attribution (classification) |
| Goal | Predict which eruption produced a sample from its chemistry. |
| Last experiment | 01_event_dummy_baseline — done |
| Last result | Brier score -1.849821 (per-class ROC AUC 0.500) |

## Data understanding

| Variable | Value |
|---|---|
| Status | done — 2026-10-31 |
| Summary | 6220 train rows and 839 test rows, 95 columns. The target is `Event`, the eruption that produced a sample: 30 classes, 567→67 rows (~8.5:1), fully observed on all 6220 training rows. 45 chemistry columns separate the sources (major oxides p ≈ 0; `Sr87_Sr86` only marginal at p = 0.064), six major oxides give 52.7% of the variance in two PCA components, and between-source variance is 15–26× the within-sample variance over 222 replicated samples. Trace elements exist for only 26 columns at ~20% coverage and are absent for three volcanoes; geographic distance barely tracks composition distance (Spearman 0.215). The shipped split is sample-grouped (0 shared `SampleID`, all 30 events in both tables), so it tests new analyses of known eruptions. 69 leakage flags come from class-conditional null rates in `Magnitude`, `VEI`, `HistoricalAge` and the location columns, and `Authors`/`DOI` cover 60.9% of rows with volcano-unique values. |
| Report | [data_analysis/data_analysis.md](../data_analysis/data_analysis.md) |

## Modeling decisions

| Variable | Value |
|---|---|
| Status | locked |
| Revised on | 2026-10-31 |
| Prediction goal | probabilities |
| Deployment | groups |
| Horizon | n/a |
| Gap | n/a |
| Generalize to | SampleID |
| Known at predict | n/a |
| Time role | n/a |
| Metric role | proper_score |
| Metric | Brier score over rows and classes plus per-class ROC AUC, with no macro or micro average across classes |
| Baseline | dummy |
| Baseline note | always predict PCC2011 |
| Folds | predefined |

## History

| Stem | Intent (one line) | Status | Headline result | Report | Design note |
|---|---|---|---|---|---|
| 01_event_dummy_baseline | Dummy reference for the 30-event source-attribution target | done | Brier score -1.849821 (per-class ROC AUC 0.500) | local workspace: [reports/](../reports/) · id: 01a1927f-a51d-76f4-814e-5a4eb30218d0<br>absolute path: /Users/glemaitre/Documents/scratch/winter-school/reports/ | [journal/01_event_dummy_baseline.md](01_event_dummy_baseline.md) |

## Ideas

| Question | Status | Experiment | Source |
|---|---|---|---|
| Does the usable feature set change the comparison once a real learner is tried? | open | 01_event_dummy_baseline | design:01_event_dummy_baseline |
| Can the report carry its training bindings so the train-versus-test checks actually run? | open | 01_event_dummy_baseline | audit:01_event_dummy_baseline:checks.SKD001 |

## Backlog

| # | Item | Source |
|---|---|---|
