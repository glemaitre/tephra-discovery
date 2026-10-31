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
| Summary | 6220 train rows and 839 test rows, 95 columns. Targets are the source labels `Volcano` (13 classes, 1685→77 rows, ~22:1) and `Event` (30 classes, 567→67, ~8.5:1); both fully observed. 45 chemistry columns separate the sources (major oxides p ≈ 0 against both targets; `Sr87_Sr86` only marginal at p = 0.021/0.064), six major oxides give 52.7% of the variance in two PCA components, and between-volcano variance is 15–26× the within-sample variance over 222 replicated samples. Trace elements exist for only 26 columns at ~20% coverage and are absent for Huanquihue Group, Quetrupillán and Villarrica; geographic distance barely tracks composition distance (Spearman 0.215). The shipped split is sample-grouped (0 shared `SampleID`, all 13 volcanoes and 30 events in both tables), so it tests new analyses of known sources. `Authors`/`DOI` cover 60.9% of rows with volcano-unique values — a leakage risk. |
| Report | [data_analysis/data_analysis.md](../data_analysis/data_analysis.md) |

## Modeling decisions

| Variable | Value |
|---|---|
| Status | <missing \| draft \| locked> |
| Revised on | <date or n/a> |
| Prediction goal | <probabilities \| point_labels \| intervals \| point_predictions> |
| Deployment | <iid \| time \| groups> |
| Horizon | <n/a, or one or more number-and-unit values> |
| Gap | <n/a or number and unit> |
| Generalize to | <n/a or column> |
| Known at predict | <n/a or columns> |
| Time role | <n/a \| sort_key \| covariate> |
| Metric role | <imposed \| proper_score \| ranking \| thresholded \| point_error> |
| Metric | <comparison name> |
| Baseline | <one of seasonal_naive, group_mean, logistic, production, dummy> |
| Baseline note | <one short phrase> |
| Folds | <integer at least 1, or predefined> |

## History

| Stem | Intent (one line) | Status | Headline result | Report | Design note |
|---|---|---|---|---|---|

## Ideas

| Question | Status | Experiment | Source |
|---|---|---|---|

## Backlog

| # | Item | Source |
|---|---|---|
