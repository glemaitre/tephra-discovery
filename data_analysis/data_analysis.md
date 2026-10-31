# Exploratory data analysis — volcanic geochemistry

## Dataset at a glance

<iframe src="data_analysis_train.html" width="100%" height="640"></iframe>

<iframe src="data_analysis_test.html" width="100%" height="640"></iframe>

## Modelling implications

- **The targets are source labels, and both are fully observed.** The
  training table has 6220 rows and 95 columns (71 float, 24 string); the test
  table has 839 rows with 73 float and 22 string columns. `Volcano` has 13
  classes with no missing values and `Event` has 30 classes with no missing
  values. The task is source attribution: assign an analysis to the volcano
  and the eruption that produced it.
- **Both targets are strongly imbalanced.** `Volcano` runs from 1685 rows
  (Mocho-Choshuenco) down to 77 (Grupo Antillanca), about 22:1. `Event` runs
  from 567 (PCC2011) down to 67 (Quet1), about 8.5:1. Accuracy alone would be
  misleading; a macro-averaged or per-class metric and a majority-class
  baseline are the relevant comparisons.
- **The chemistry separates the sources, and the strongest signal is silica.**
  Of 45 chemistry columns, the raw major oxides `SiO2`, `K2O`, `MgO`, `FeOT`
  and `Al2O3` all return p ≈ 0 against both targets, and the trace elements
  and ratios follow (`Zr` p = 1.7e-222, `La_Yb` p = 2.9e-215, `Rb` p =
  6.1e-201 against `Volcano`). The isotope ratio `Sr87_Sr86` is the
  exception: p = 0.021 against `Volcano` and p = 0.064 against `Event`, so it
  is only marginally informative.
- **Six major oxides carry much of the structure.** A pairplot of the six
  best-covered chemistry columns (`SiO2`, `CaO`, `K2O`, `Na2O`, `Al2O3`,
  `TiO2`) colours the sources two at a time.

  ![Pairplot of the six best-covered major oxides, coloured by Volcano](interactions.png)

  The same six columns dominate the PCA projection of all 45 chemistry
  columns: two components retain 52.7% of the variance (PC1 37.9%, PC2 14.8%).

  ![Two-component PCA of the 45 chemistry columns, coloured by Volcano](pca.png)

- **Major oxides separate most volcano pairs, but the trace block is thin.**
  The closest pairs in standardised major-oxide space are Michinmahuida and
  Quetrupillán (distance 1.03), Chaitén and Reclus (1.06), Monte Burney and
  Reclus (1.21), and Puyehue-Cordón Caulle and Quetrupillán (1.31). Trace
  elements exist for only 26 columns at about 20% coverage, and three
  volcanoes — Huanquihue Group, Quetrupillán and Villarrica — have no trace
  data at all, so the trace block can only arbitrate 45 of the 78 volcano
  pairs. Where it does apply it changes the picture: Chaitén and Reclus sit
  1.06 apart on majors but 5.82 apart on traces.

  ![Median trace and rare-earth concentrations per volcano, log scale](trace_element_patterns.png)

- **The major oxides are not independent of silica.** Against SiO2 the other
  oxides follow the differentiation trends the literature reports, so several
  columns carry overlapping information.

  ![Major oxides against SiO2, coloured by source](oxide_variation.png)

  ![Total-alkali-versus-silica position per analysis](tas_diagram.png)

- **Replicate noise is small next to source differences.** Among the 222
  samples that were analysed more than once, the between-volcano variance is
  25.7× the within-sample variance for `FeOT`, 25.4× for `SiO2`, 19.1× for
  `Zr_Nb`, 17.5× for `CaO` and 15.2× for `TiO2`. This is what makes source
  attribution feasible on this table, and it also sets the noise ceiling a
  model has to beat.
- **No column is dominated by a reporting floor.** The largest single-value
  share is 11.1% for `Sr87_Sr86`, which has only 9 values; among the
  well-populated trace columns the repeated-value share reaches 34.2% (`Tb`)
  while no single value exceeds 2%. That pattern is consistent with rounding
  at a few significant figures rather than censoring.
- **Technique and source are confounded.** Median `TiO2` is 0.70 under EMP
  but 1.87 under LA-ICP-MS, and median `Dy` is 5.69 against 3.82. The
  technique split is itself very uneven — EMP covers 4790 rows across all 13
  volcanoes, LA-ICP-MS 1092 rows across 7 — so an element that differs by
  technique may only be showing which volcanoes that technique sampled.
- **Geographic distance barely predicts compositional distance.** Across all
  78 volcano pairs the Spearman correlation between great-circle distance and
  major-oxide distance is 0.215, and across the 45 pairs with trace data it is
  0.100. The closest pair geographically, Quetrupillán and Villarrica at
  13.6 km, is 4.29 apart on majors. Neighbouring sources are therefore not
  automatically the confusable ones here.
- **Rows are far from independent.** Mocho-Choshuenco contributes 1685 rows
  from only 90 distinct `SampleID` values across 8 events; a sample is
  measured a median of 8 times, up to 70. A random row split would put the
  same sample on both sides of the evaluation.
- **The shipped split is already grouped by sample.** Train has 455 distinct
  `SampleID` values and test has 118, with zero shared keys, while all 13
  volcanoes and all 30 events appear on both sides. The test set therefore
  measures unseen *analyses of known sources*, not unseen sources.
- **Analytical provenance leaks the source.** `Authors` and `DOI` have 25
  values each, 18 of which occur for exactly one volcano, covering 60.9% of
  rows; `MeasurementRun` has 34 of 52 values confined to one volcano (27.7%
  of rows). `AnalyzedMaterial` (9 of 12 values, 2.7% of rows) and
  `AnalyticalTechnique` (5 of 12, 1.3%) are milder.
- **Size and age columns behave as event constants.** `VEI` is missing for
  59.8% of rows (3721 of 6220) and takes four values: VEI 3 on 144 rows, 4 on
  528, 5 on 1819, 6 on 8. `Magnitude` is missing for 46.2% and spans 13
  values. `HistoricalAge` is 80.1% missing but present only for `1852
  Llaima`, `PCC2011` and `Vcha-2008`. `RadiocarbonAge` is entirely empty and
  `Ar40Ar39_Age` has two rows.
- **Two arithmetic identities hold in the chemistry block.** `Total` equals
  the sum of the major oxides to a median absolute gap of 1.4e-14 (the
  maximum gap, 14.7, comes from rows with partially missing oxides), and each
  `*_normalized` column equals `oxide / Total × 100` to about 3e-10. They are
  deterministic duplicates, so they are excluded from the 45 modelling
  columns.
- **The coordinates are clean and regional.** `Latitude` spans -53.75 to
  -38.7 and `Longitude` -75.68 to -68.7, both with zero missing values and no
  out-of-range points — the southern Andes. A model trained here will not
  transfer to another arc.
- **Duplicates: zero** exact duplicate rows in either table.

## Open questions

- **Which of the two targets first?** `Volcano` has fewer classes and more
  rows per class; `Event` is finer-grained and harder. A model is reported
  for one of them, not both jointly.
- **Is the shipped split the evaluation we want?** It already separates
  samples but keeps every volcano and every event in training. Reporting on
  it measures new analyses of known sources; a model that must generalise to
  a new volcano needs a grouped split instead.
- **Do the provenance columns belong in the model at all?** `Authors`, `DOI`
  and `MeasurementRun` would have to be excluded or grouped rather than
  encoded, because they would let a model read the answer.
- **Should the oxide block be transformed?** Major-element data are
  compositional, and the Harker views show the oxides are mutually dependent,
  so a log-ratio representation is the candidate preprocessing before any
  linear or distance-based model.
- **What should happen to the three trace-free volcanoes?** Huanquihue Group,
  Quetrupillán and Villarrica cannot be arbitrated by trace elements, so a
  trace-based comparison silently drops them.
- **What would a new sample look like?** A model that uses `Authors`, `DOI`,
  `Flag` or event-level size and age attributes could not score an
  unlabelled sample, since those fields are filled in during curation.
