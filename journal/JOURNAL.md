# JOURNAL

## Status

| Variable | Value |
|---|---|
| Project / dataset | <fill in — e.g., `adult-census` classification> |
| Goal | <one sentence — what would "done" look like for this project?> |
| Last experiment | <NN_name> — <status: planned \| approved \| running \| done \| abandoned> |
| Last result | <one-line headline metric, or "n/a" if not yet run> |

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
| Metric | Brier score plus ROC AUC |
| Baseline | dummy |
| Baseline note | always predict PCC2011 |
| Folds | predefined |

## History

| Stem | Intent (one line) | Status | Headline result | Report | Design note |
|---|---|---|---|---|---|

## Ideas

| Question | Status | Experiment | Source |
|---|---|---|---|

## Backlog

| # | Item | Source |
|---|---|---|
