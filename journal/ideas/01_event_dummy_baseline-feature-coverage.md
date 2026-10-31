# Feature coverage before a real learner

- **Experiment:** 01_event_dummy_baseline
- **Source:** design:01_event_dummy_baseline
- **Triage:** open

## Question

Does the usable feature set change the comparison once a real learner is
tried? The exploration counted 45 chemistry columns, but the trace elements
(including the REE block and the `La_Yb` and `Zr_Nb` ratios) cover only about
20% of rows, and `Sr87_Sr86` has 9 non-null values in total.

## Why now

The baseline is a dummy that ignores its features, so it cannot show what the
feature matrix actually contains. Before any real learner is compared against
it, the columns that are mostly missing need a decision — dropped, imputed,
or kept with a missingness indicator — otherwise the comparison silently
depends on how many rows happen to carry trace data.

## What changes

Measure the per-column coverage and the row count that survives a complete-case
requirement, then pick a missing-data strategy for the trace block: drop the
columns below a coverage floor, or median-impute with a per-column missing
indicator. Re-run the same locked comparison afterwards.

## Open gaps

- No coverage floor is chosen yet; the exploration reported per-column
  missingness but not the joint row survival.
- It is unknown whether the trace-carrying rows are spread across all 30
  events or concentrated in a few, which decides whether imputation is honest.
