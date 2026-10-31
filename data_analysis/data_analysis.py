# %% [markdown]
# # Exploratory data analysis: volcanic geochemistry
#
# Overview of the volcanic geochemistry tables (`data/train.csv` and
# `data/test.csv`) before choosing a learner and splitter. TableReport owns
# dtypes, missingness, univariate distributions, cardinality, and top
# pairwise associations — do not re-plot those.
#
# The prediction target is `Event`: the eruption that produced a sample, so
# the task is **source attribution**. `Volcano`, the source edifice, is the
# coarser label and is the subject of a separate later experiment.

# %%
import pandas as pd
import seaborn as sns
import skrub

from winter_school import PROJECT_ROOT

OUT = PROJECT_ROOT / "data_analysis"
OUT.mkdir(parents=True, exist_ok=True)

RAW = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
FRAME = RAW.to_pandas() if hasattr(RAW, "to_pandas") else RAW
RAW

# %% [markdown]
# ## Target distribution
#
# `Event` is the eruption that produced a sample. It is fully observed in the
# training table; class counts follow.

# %%
TARGET = "Event"
y = FRAME[TARGET]
counts = y.value_counts(dropna=False)
target_summary = counts.rename("count").to_frame()
target_summary["rate"] = target_summary["count"] / target_summary["count"].sum()
target_summary

# %%
plot_df = target_summary.reset_index(names=TARGET)
g = sns.catplot(data=plot_df, x=TARGET, y="count", kind="bar")
g.savefig(OUT / "target_distribution.png", bbox_inches="tight")
g

# %% [markdown]
# ## Table overview: train
#
# Column types, missingness, cardinality, and pairwise associations.

# %%
report = skrub.TableReport(RAW, title="train", verbose=0)
report.write_html(OUT / "data_analysis_train.html")
report

# %% [markdown]
# ## Duplicate rows: train
#
# Exact duplicates inflate counts and can leak across a later split.
# Cleaning belongs in the pipeline, not by dropping rows here.

# %%
n_dup = int(FRAME.duplicated().sum())
duplicate_summary = pd.DataFrame(
    {
        "n_rows": [len(FRAME)],
        "n_duplicate_rows": [n_dup],
        "duplicate_rate": [n_dup / len(FRAME) if len(FRAME) else 0.0],
    }
)
duplicate_summary

# %% [markdown]
# ## Table overview: test
#
# Another table in this dataset.

# %%
RAW_test = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")
FRAME_test = RAW_test.to_pandas() if hasattr(RAW_test, "to_pandas") else RAW_test
RAW_test

# %%
report_test = skrub.TableReport(RAW_test, title="test", verbose=0)
report_test.write_html(OUT / "data_analysis_test.html")
report_test

# %% [markdown]
# ## Duplicate rows: test
#
# Cleaning belongs in the pipeline, not by dropping rows here.

# %%
n_dup_test = int(FRAME_test.duplicated().sum())
duplicate_summary_test = pd.DataFrame(
    {
        "n_rows": [len(FRAME_test)],
        "n_duplicate_rows": [n_dup_test],
        "duplicate_rate": [n_dup_test / len(FRAME_test) if len(FRAME_test) else 0.0],
    }
)
duplicate_summary_test

# %% [markdown]
# ## Candidate targets: availability
#
# No target is fixed yet. The plausible prediction targets on this table are
# the source labels (`Volcano`, `Event`), the deposit-size labels (`VEI`,
# `Magnitude`) and the age labels. This cell shows how well each is observed
# in the training table before one is chosen.

# %%
CANDIDATE_TARGETS = [
    "Volcano",
    "Event",
    "VEI",
    "Magnitude",
    "HistoricalAge",
    "RadiocarbonAge",
    "Ar40Ar39_Age",
]
candidate_summary = pd.DataFrame(
    {
        "dtype": [
            str(FRAME[c].dtype) if c in FRAME.columns else "absent"
            for c in CANDIDATE_TARGETS
        ],
        "n_missing": [
            int(FRAME[c].isna().sum()) if c in FRAME.columns else None
            for c in CANDIDATE_TARGETS
        ],
        "missing_rate": [
            float(FRAME[c].isna().mean()) if c in FRAME.columns else None
            for c in CANDIDATE_TARGETS
        ],
        "n_unique": [
            int(FRAME[c].nunique(dropna=True)) if c in FRAME.columns else None
            for c in CANDIDATE_TARGETS
        ],
    },
    index=CANDIDATE_TARGETS,
)
candidate_summary

# %% [markdown]
# ## Rows per source volcano
#
# Each volcano contributes many rows but far fewer distinct samples, and the
# counts are uneven. Rows are therefore not independent, which matters for
# both the target choice and any later split.

# %%
rows_per_source = pd.DataFrame(
    {
        "n_rows": FRAME.groupby("Volcano", dropna=False).size(),
        "n_events": FRAME.groupby("Volcano", dropna=False)["Event"].nunique(),
        "n_samples": FRAME.groupby("Volcano", dropna=False)["SampleID"].nunique(),
    }
).sort_values("n_rows", ascending=False)
rows_per_source

# %% [markdown]
# ## Analytical metadata against the source label
#
# If a study, a laboratory run or an analytical technique appears for only one
# volcano, then that column already carries the source label and cannot be
# used as a feature. This cell measures how concentrated each metadata column
# is within a single volcano.

# %%
METADATA_COLUMNS = [
    "Authors",
    "DOI",
    "AnalyticalTechnique",
    "AnalyzedMaterial",
    "MeasurementRun",
    "TypeOfAnalysis",
]
leakage_rows = []
for column in METADATA_COLUMNS:
    if column not in FRAME.columns:
        continue
    value_to_volcanoes = (
        FRAME.dropna(subset=[column]).groupby(column, dropna=False)["Volcano"].nunique()
    )
    single_volcano = value_to_volcanoes[value_to_volcanoes <= 1]
    rows_in_single = int(FRAME[column].isin(single_volcano.index).sum())
    leakage_rows.append(
        {
            "column": column,
            "n_values": int(FRAME[column].nunique(dropna=True)),
            "n_values_in_one_volcano": int(len(single_volcano)),
            "share_rows_in_one_volcano": rows_in_single / len(FRAME),
        }
    )
metadata_leakage = pd.DataFrame(leakage_rows).set_index("column")
metadata_leakage

# %% [markdown]
# ## Chemistry identities
#
# `Total` is reported next to the major oxides, and the `*_normalized`
# columns are derived from it. Confirming the arithmetic keeps a model from
# learning an identity instead of a signal.

# %%
OXIDES = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeO",
    "Fe2O3",
    "FeOT",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
    "Cl",
    "LOI",
]
present_oxides = [c for c in OXIDES if c in FRAME.columns]
oxide_sum = FRAME[present_oxides].sum(axis=1, skipna=True, min_count=1)
total_gap = (oxide_sum - FRAME["Total"]).abs()
identity_summary = pd.DataFrame(
    {
        "n_rows_with_oxides": [int(oxide_sum.notna().sum())],
        "median_abs_gap_vs_Total": [float(total_gap.median())],
        "max_abs_gap_vs_Total": [float(total_gap.max())],
    }
)
identity_summary

# %%
NORMALIZED_PAIRS = [
    ("SiO2", "SiO2_normalized"),
    ("TiO2", "TiO2_normalized"),
    ("Al2O3", "Al2O3_normalized"),
    ("FeOT", "FeOT_normalized"),
    ("MnO", "MnO_normalized"),
    ("MgO", "MgO_normalized"),
    ("CaO", "CaO_normalized"),
    ("Na2O", "Na2O_normalized"),
    ("K2O", "K2O_normalized"),
    ("P2O5", "P2O5_normalized"),
    ("Cl", "Cl_normalized"),
]
normalized_rows = []
for raw_column, normalized_column in NORMALIZED_PAIRS:
    if raw_column in FRAME.columns and normalized_column in FRAME.columns:
        implied = FRAME[raw_column] / FRAME["Total"] * 100.0
        gap = (implied - FRAME[normalized_column]).abs()
        normalized_rows.append(
            {"raw": raw_column, "median_abs_gap": float(gap.median())}
        )
normalized_check = pd.DataFrame(normalized_rows).set_index("raw")
normalized_check

# %% [markdown]
# ## Eruption-size label coverage
#
# `VEI` and `Magnitude` are attributes of an event that repeat for every
# sample of that event. This cell shows how many samples and volcanoes
# actually carry a size label, and the spread of the observed `VEI` values.

# %%
size_summary = pd.DataFrame(
    {
        "n_missing": [
            int(FRAME["VEI"].isna().sum()),
            int(FRAME["Magnitude"].isna().sum()),
        ],
        "missing_rate": [
            float(FRAME["VEI"].isna().mean()),
            float(FRAME["Magnitude"].isna().mean()),
        ],
        "n_volcanoes_covered": [
            int(FRAME.loc[FRAME["VEI"].notna(), "Volcano"].nunique()),
            int(FRAME.loc[FRAME["Magnitude"].notna(), "Volcano"].nunique()),
        ],
        "n_events_covered": [
            int(FRAME.loc[FRAME["VEI"].notna(), "Event"].nunique()),
            int(FRAME.loc[FRAME["Magnitude"].notna(), "Event"].nunique()),
        ],
    },
    index=["VEI", "Magnitude"],
)
size_summary

# %%
FRAME["VEI"].value_counts(dropna=False).sort_index()

# %% [markdown]
# ## The chemistry block
#
# The numeric chemistry columns are the raw oxides and trace elements plus a
# few element ratios. The `*_normalized` duplicates and the analytical
# standard errors are excluded so that no column counts twice.

# %%
NON_CHEMISTRY_COLUMNS = {
    "Volcano",
    "Event",
    "Magnitude",
    "VEI",
    "SampleID",
    "SampleObservationID",
    "IGSN",
    "Location",
    "Latitude",
    "Longitude",
    "Authors",
    "DOI",
    "TypeOfRegister",
    "TypeOfAnalysis",
    "AnalyzedMaterial",
    "AnalyticalTechnique",
    "TypeOfSection",
    "SectionID",
    "SubSectionID",
    "SubSection_DistanceFromTop_cm",
    "HistoricalAge",
    "RadiocarbonAge",
    "RadiocarbonAgeError",
    "RadiocarbonLabCode",
    "StratigraphicPosition",
    "Ar40Ar39_Age",
    "Ar40Ar39_Age_Error",
    "DepositColor",
    "DepositThickness_cm",
    "GrainSize_min",
    "GrainSize_max",
    "MeasurementRun",
    "Comments",
    "Flag",
    "FlagDescription",
}
CHEM_COLUMNS = [
    column
    for column in FRAME.columns
    if column not in NON_CHEMISTRY_COLUMNS
    and not column.endswith("_normalized")
    and column != "Total_normalization"
    and column != "Total"
    and not column.endswith("_2SE")
]
chemistry_summary = pd.DataFrame(
    {
        "n_chemistry_columns": [len(CHEM_COLUMNS)],
        "n_columns_fully_null": [int(FRAME[CHEM_COLUMNS].isna().all().sum())],
        "median_coverage": [float(FRAME[CHEM_COLUMNS].notna().mean().median())],
    }
)
chemistry_summary

# %% [markdown]
# ## Join keys and coverage: train vs test
#
# The two tables share a schema. This cell checks which columns are shared and
# whether the same physical sample appears in both, which decides whether the
# shipped split is safe for a source target.

# %%
left = FRAME
right = FRAME_test
shared = sorted(left.columns.intersection(right.columns))
join_key_summary = pd.DataFrame(
    {
        "column": shared,
        "left_dtype": [str(left[c].dtype) for c in shared],
        "right_dtype": [str(right[c].dtype) for c in shared],
        "left_nunique": [int(left[c].nunique(dropna=False)) for c in shared],
        "right_nunique": [int(right[c].nunique(dropna=False)) for c in shared],
        "left_duplicated_key": [int(left[c].duplicated().sum()) for c in shared],
        "right_duplicated_key": [int(right[c].duplicated().sum()) for c in shared],
    }
)
join_key_summary.head(20)

# %%
KEY = "SampleID"
left_key = left[KEY].dropna()
right_key = right[KEY].dropna()
left_set = set(left_key.tolist())
right_set = set(right_key.tolist())
join_coverage = pd.DataFrame(
    {
        "metric": [
            "left_rows",
            "right_rows",
            "left_only_keys",
            "right_only_keys",
            "shared_keys",
        ],
        "value": [
            len(left),
            len(right),
            len(left_set - right_set),
            len(right_set - left_set),
            len(left_set & right_set),
        ],
    }
)
join_coverage

# %% [markdown]
# ## Text columns
#
# `Authors`, `DOI`, `Comments` and `FlagDescription` carry curated metadata.
# Their length and empty rate show how much per-row information they hold.

# %%
TEXT_COLUMNS = ["Authors", "DOI", "Comments", "FlagDescription"]
text_rows = []
for column in TEXT_COLUMNS:
    if column not in FRAME.columns:
        continue
    series = FRAME[column].astype("string")
    non_null = series.dropna()
    text_rows.append(
        {
            "column": column,
            "n_non_null": int(non_null.shape[0]),
            "n_unique": int(non_null.nunique()),
            "empty_string_rate": (
                float((non_null == "").mean()) if len(non_null) else None
            ),
            "mean_length": (
                float(non_null.str.len().mean()) if len(non_null) else None
            ),
            "max_length": (int(non_null.str.len().max()) if len(non_null) else None),
        }
    )
text_summary = pd.DataFrame(text_rows).set_index("column")
text_summary

# %% [markdown]
# ## Geo columns
#
# Latitude and Longitude locate each sample. This cell checks their ranges
# and counts coordinates outside the plausible global envelope.

# %%
GEO_COLUMNS = ["Latitude", "Longitude"]
geo_rows = []
for column in GEO_COLUMNS:
    series = FRAME[column]
    observed = series.dropna()
    if column == "Latitude":
        out_of_range = ((observed < -90) | (observed > 90)).sum()
    else:
        out_of_range = ((observed < -180) | (observed > 180)).sum()
    geo_rows.append(
        {
            "column": column,
            "n_missing": int(series.isna().sum()),
            "min": float(observed.min()) if len(observed) else None,
            "max": float(observed.max()) if len(observed) else None,
            "n_out_of_range": int(out_of_range),
        }
    )
geo_summary = pd.DataFrame(geo_rows).set_index("column")
geo_summary

# %% [markdown]
# ## Subgroup: analytical technique against the source targets
#
# Grouping by `AnalyticalTechnique` shows whether the measurement method
# itself separates the volcanoes and events — the leakage question read from
# the target side.

# %%
subgroup_summary = (
    FRAME.groupby("AnalyticalTechnique", dropna=False)
    .agg(
        n_rows=("Volcano", "size"),
        n_volcanoes=("Volcano", "nunique"),
        n_events=("Event", "nunique"),
    )
    .sort_values("n_rows", ascending=False)
)
subgroup_summary

# %% [markdown]
# ## Hypothesis tests: chemistry against each target
#
# A numeric chemistry column gets a Kruskal–Wallis test across the classes of
# `Volcano` and of `Event`; a categorical column gets a chi-square test on its
# crosstab with the target. The p-values are a signal, not a modelling
# decision.

# %%
from scipy import stats

TEST_COLUMNS = [
    "SiO2",
    "Al2O3",
    "FeOT",
    "MgO",
    "K2O",
    "Rb",
    "Sr",
    "Zr",
    "La_Yb",
    "Sr87_Sr86",
]
CATEGORICAL_TEST_COLUMNS = [
    "AnalyticalTechnique",
    "AnalyzedMaterial",
    "TypeOfAnalysis",
]
hypothesis_rows = []
for target in ["Volcano", "Event"]:
    for column in TEST_COLUMNS:
        subset = FRAME[[column, target]].dropna()
        groups = [values.to_numpy() for _, values in subset.groupby(target)[column]]
        groups = [group for group in groups if len(group) > 1]
        if len(groups) < 2:
            continue
        statistic, p_value = stats.kruskal(*groups)
        hypothesis_rows.append(
            {
                "target": target,
                "feature": column,
                "test": "kruskal",
                "n_groups": len(groups),
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    for column in CATEGORICAL_TEST_COLUMNS:
        subset = FRAME[[column, target]].dropna()
        crosstab = pd.crosstab(subset[column], subset[target])
        if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
            continue
        statistic, p_value, _dof, _expected = stats.chi2_contingency(crosstab)
        hypothesis_rows.append(
            {
                "target": target,
                "feature": column,
                "test": "chi2",
                "n_groups": int(crosstab.shape[0]),
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
hypothesis_tests = pd.DataFrame(hypothesis_rows).sort_values("p_value")
hypothesis_tests

# %% [markdown]
# ## Interactions between chemistry columns
#
# A pairplot on six of the best-covered chemistry columns, coloured by
# `Volcano`, shows whether the sources separate two at a time. Values are
# median-filled for the plot only.

# %%
coverage = FRAME[CHEM_COLUMNS].notna().mean().sort_values(ascending=False)
top_features = coverage.head(6).index.tolist()
plot_frame = FRAME[top_features + ["Volcano"]].copy()
plot_frame[top_features] = plot_frame[top_features].fillna(
    plot_frame[top_features].median()
)
g = sns.pairplot(plot_frame, hue="Volcano", corner=True, diag_kind="hist", height=2.2)
g.figure.savefig(OUT / "interactions.png", dpi=120)
g

# %% [markdown]
# ## PCA projection of the chemistry block
#
# Two principal components of the median-imputed chemistry columns, coloured
# by `Volcano`. This is a view for exploration only; it is not a model
# preprocessing step.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_frame = FRAME[CHEM_COLUMNS].copy()
pca_frame = pca_frame.fillna(pca_frame.median())
pca_frame = pca_frame.dropna(axis=1, how="all")
scaled = StandardScaler().fit_transform(pca_frame)
components = PCA(n_components=2, random_state=0).fit_transform(scaled)
pca_plot = pd.DataFrame(
    {
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Volcano": FRAME["Volcano"].values,
    }
)
g = sns.relplot(data=pca_plot, x="PC1", y="PC2", hue="Volcano", height=6, aspect=1.2)
g.figure.savefig(OUT / "pca.png", dpi=120)
g

# %% [markdown]
# ## Major-oxide versus trace-element separability
#
# The tephra literature reports that major oxides separate most sources but
# not all, and that trace elements resolve the remainder. This table ranks the
# volcano pairs that sit closest in each space and shows whether the ranking
# changes.

# %%
import numpy as np

MAJOR_COLUMNS = ["SiO2", "TiO2", "Al2O3", "CaO", "Na2O", "K2O"]
TRACE_COLUMNS = [
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "Pb",
    "Th",
    "U",
]


def _standardised_centroids(columns, min_coverage=0.2):
    coverage = FRAME[columns].notna().mean()
    kept = [column for column in columns if coverage[column] >= min_coverage]
    values = FRAME[kept]
    scaled = (values - values.mean()) / values.std(ddof=0)
    centroids = scaled.assign(Volcano=FRAME["Volcano"].values).groupby("Volcano").mean()
    return centroids, kept


def _centroid_distance(left_centroid, right_centroid, min_shared=5):
    shared = left_centroid.notna() & right_centroid.notna()
    if int(shared.sum()) < min_shared:
        return float("nan")
    delta = left_centroid[shared] - right_centroid[shared]
    return float(np.linalg.norm(delta))


major_centroids, major_kept = _standardised_centroids(MAJOR_COLUMNS)
trace_centroids, trace_kept = _standardised_centroids(TRACE_COLUMNS)
separability_rows = []
volcano_names = major_centroids.index.tolist()
for index, left_name in enumerate(volcano_names):
    for right_name in volcano_names[index + 1 :]:
        separability_rows.append(
            {
                "left": left_name,
                "right": right_name,
                "major_distance": _centroid_distance(
                    major_centroids.loc[left_name], major_centroids.loc[right_name]
                ),
                "trace_distance": _centroid_distance(
                    trace_centroids.loc[left_name], trace_centroids.loc[right_name]
                ),
            }
        )
separability = pd.DataFrame(separability_rows).sort_values("major_distance")
separability.head(10)

# %% [markdown]
# ## Trace-element and rare-earth patterns
#
# Median trace-element concentrations per volcano on a log scale, in the order
# the columns appear in the table. The literature uses these secondary
# fingerprints once the major oxides overlap.

# %%
pattern_medians = (
    FRAME.groupby("Volcano")[TRACE_COLUMNS].median().where(lambda frame: frame > 0)
)
pattern_long = (
    pattern_medians.reset_index()
    .melt(id_vars="Volcano", var_name="element", value_name="median")
    .dropna()
)
pattern_long["log10_median"] = np.log10(pattern_long["median"])
g = sns.relplot(
    data=pattern_long,
    x="element",
    y="log10_median",
    hue="Volcano",
    kind="line",
    height=6,
    aspect=1.6,
)
g.figure.savefig(OUT / "trace_element_patterns.png", dpi=120)
g

# %% [markdown]
# ## Oxide variation against silica
#
# Harker-style views of the other major oxides against SiO2, coloured by
# source. Oxides that track silica closely are partly redundant with it.

# %%
VARIATION_OXIDES = [
    "TiO2",
    "Al2O3",
    "FeOT",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
]
variation_long = (
    FRAME[["Volcano", "SiO2"] + VARIATION_OXIDES]
    .melt(id_vars=["Volcano", "SiO2"], var_name="oxide", value_name="oxide_value")
    .dropna()
)
g = sns.relplot(
    data=variation_long,
    x="SiO2",
    y="oxide_value",
    col="oxide",
    col_wrap=3,
    hue="Volcano",
    kind="scatter",
    height=2.6,
    facet_kws={"sharey": False},
)
g.figure.savefig(OUT / "oxide_variation.png", dpi=120)
g

# %% [markdown]
# ## Total-alkali-versus-silica position
#
# The IUGS classification position of each analysis, using the normalised
# columns already present in the table.

# %%
tas_frame = pd.DataFrame(
    {
        "SiO2": FRAME["SiO2_normalized"],
        "alkalis": FRAME["Na2O_normalized"] + FRAME["K2O_normalized"],
        "Volcano": FRAME["Volcano"],
    }
).dropna()
g = sns.relplot(
    data=tas_frame, x="SiO2", y="alkalis", hue="Volcano", height=6, aspect=1.2
)
g.figure.savefig(OUT / "tas_diagram.png", dpi=120)
g

# %% [markdown]
# ## Within-sample versus between-source variance
#
# A sample is measured several times. Comparing the variance of replicate
# analyses against the variance of volcano means gives a noise ceiling and
# shows which elements are precise enough to separate sources.

# %%
variance_rows = []
for column in CHEM_COLUMNS:
    subset = FRAME[[column, "SampleID", "Volcano"]].dropna()
    if subset.empty:
        continue
    grouped = subset.groupby("SampleID")[column]
    sample_sizes = grouped.size()
    replicated = sample_sizes[sample_sizes >= 2].index
    if len(replicated) < 2:
        continue
    within = (
        subset[subset["SampleID"].isin(replicated)]
        .groupby("SampleID")[column]
        .var(ddof=0)
        .mean()
    )
    between = subset.groupby("Volcano")[column].mean().var(ddof=0)
    if not within or not np.isfinite(within):
        continue
    variance_rows.append(
        {
            "column": column,
            "n_replicated_samples": int(len(replicated)),
            "within_sample_variance": float(within),
            "between_volcano_variance": float(between),
            "between_over_within": float(between / within),
        }
    )
variance_decomposition = (
    pd.DataFrame(variance_rows)
    .set_index("column")
    .sort_values("between_over_within", ascending=False)
)
variance_decomposition.head(15)

# %% [markdown]
# ## Value floors and repeated values
#
# Analytical floors and reported rounding leave trails in the raw values. A
# high share of repeated or modal values marks a column whose variation is
# partly censoring rather than measurement.

# %%
floor_rows = []
for column in CHEM_COLUMNS:
    values = FRAME[column].dropna()
    if values.empty:
        continue
    counts = values.value_counts()
    floor_rows.append(
        {
            "column": column,
            "n_values": int(len(values)),
            "n_distinct": int(values.nunique()),
            "repeated_value_share": float(counts[counts > 1].sum() / len(values)),
            "modal_value_share": float(counts.iloc[0] / len(values)),
        }
    )
value_floor = (
    pd.DataFrame(floor_rows)
    .set_index("column")
    .sort_values("modal_value_share", ascending=False)
)
value_floor.head(15)

# %% [markdown]
# ## Technique-sensitive columns
#
# The literature reports that TiO2, MnO, P2O5 and the middle-to-heavy rare
# earths are the elements most affected by analytical artefacts. Their median
# value per technique shows whether the instrument shifts the concentration.

# %%
SENSITIVE_COLUMNS = [
    "TiO2",
    "MnO",
    "P2O5",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]
technique_sensitivity = FRAME.groupby("AnalyticalTechnique")[SENSITIVE_COLUMNS].median()
technique_sensitivity

# %% [markdown]
# ## Source proximity
#
# Geographic distance between volcano centroids against composition distance.
# Regional studies find that neighbouring centres are the confusable ones.

# %%
coords = FRAME.groupby("Volcano")[["Latitude", "Longitude"]].median()
proximity_rows = []
for index, left_name in enumerate(volcano_names):
    for right_name in volcano_names[index + 1 :]:
        lat1, lon1 = np.radians(coords.loc[left_name])
        lat2, lon2 = np.radians(coords.loc[right_name])
        delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        )
        proximity_rows.append(
            {
                "left": left_name,
                "right": right_name,
                "geographic_km": float(2 * 6371 * np.arcsin(np.sqrt(haversine))),
                "major_distance": _centroid_distance(
                    major_centroids.loc[left_name], major_centroids.loc[right_name]
                ),
                "trace_distance": _centroid_distance(
                    trace_centroids.loc[left_name], trace_centroids.loc[right_name]
                ),
            }
        )
source_proximity = pd.DataFrame(proximity_rows)
source_proximity.sort_values("geographic_km").head(10)

# %%
trace_pairs = source_proximity.dropna(subset=["trace_distance"])
source_distance_correlation = pd.DataFrame(
    {
        "space": ["major_oxides", "trace_elements"],
        "n_pairs": [int(len(source_proximity)), int(len(trace_pairs))],
        "spearman_vs_geographic_km": [
            float(
                stats.spearmanr(
                    source_proximity["geographic_km"],
                    source_proximity["major_distance"],
                ).statistic
            ),
            float(
                stats.spearmanr(
                    trace_pairs["geographic_km"],
                    trace_pairs["trace_distance"],
                ).statistic
            ),
        ],
    }
)
volcanoes_without_trace = sorted(
    set(volcano_names)
    - set(FRAME.loc[FRAME[TRACE_COLUMNS].notna().any(axis=1), "Volcano"])
)
source_distance_correlation

# %% [markdown]
# ## Feature vs target: Event
#
# A readable subset of the numeric chemistry columns against the target, not
# every pairwise plot. Identifier and provenance columns are left out because
# they would carry the label rather than predict it.

# %%
BIVARIATE_CAP = 12
bivariate_cols = [column for column in CHEM_COLUMNS if column != TARGET][:BIVARIATE_CAP]
pd.Series(bivariate_cols, name="bivariate_columns")

# %%
long = FRAME.melt(
    id_vars=[TARGET],
    value_vars=bivariate_cols,
    var_name="feature",
    value_name="value",
)
g = sns.catplot(
    data=long,
    x=TARGET,
    y="value",
    col="feature",
    col_wrap=4,
    kind="box",
    sharey=False,
)
g.savefig(OUT / "bivariate_grid.png", bbox_inches="tight")
g

# %% [markdown]
# ## Leakage candidates
#
# Flags only — not a train/test split.

# %%
leakage_flags: list[str] = []
if pd.api.types.is_numeric_dtype(y):
    corr = FRAME.select_dtypes("number").corrwith(y).abs().sort_values(ascending=False)
    for col, value in corr.items():
        if col == TARGET:
            continue
        if value is not None and value >= 0.99:
            leakage_flags.append(f"{col} abs(Pearson) vs {TARGET} = {value:.3f}")
for col in FRAME.columns:
    if col == TARGET:
        continue
    rates = FRAME.groupby(y, dropna=False)[col].apply(lambda s: s.isna().mean())
    if rates.max() - rates.min() >= 0.5:
        leakage_flags.append(f"{col} null rate differs by class: {rates.to_dict()}")
leakage_summary = pd.DataFrame({"flag": leakage_flags or ["none"]})
leakage_summary
