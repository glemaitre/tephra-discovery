# Make the train-side checks runnable

- **Experiment:** 01_event_dummy_baseline
- **Source:** audit:01_event_dummy_baseline:checks.SKD001
- **Triage:** open

## Question

Can the report carry its training bindings so the train-versus-test checks
actually run? The audit found 0 issues and 0 tips, but 16 of the automated
checks were `not applicable`, and for eight of them the reason was
"Train data is unavailable".

## Why now

SKD001 (overfitting), SKD002 (underfitting), SKD005 (underrepresented
classes), SKD008 (correlated features), SKD009 (worse than the boosting
baseline), SKD010 (slower than baseline), SKD011 (golden feature) and SKD013
(train-test overlap) all need the training data to compare against the test
data. With no training bindings the report cannot separate "no problem" from
"never checked", so the clean checks summary overstates the evidence.

## What changes

Bind the training table to the report as well as the test table, so the report
holds both datasets. Then re-run the same evaluation and read SKD001 and
SKD002 as intended: compare the train and test scores, and treat a large gap
as the overfitting signal the check documents.

## Open gaps

- The locked split is `prefit`, so the training table is currently bound only
  by the experiment's own `fit`, never handed to the report.
- Whether a `prefit` report can be constructed with training bindings at all
  is unverified; the local store round-trip needed a fix to reload such a
  report, which suggests this path is lightly travelled.
