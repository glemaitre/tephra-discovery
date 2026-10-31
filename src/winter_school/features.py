"""Feature selection for the volcanic geochemistry tables.

Owns: the stateless function that turns the loaded frame into the
feature matrix. Composition into the learner happens in `pipeline.py`.
"""

from __future__ import annotations

import pandas as pd

from winter_school.data import TARGET_COL

# Columns the exploration kept out of the model. Identifiers
# (`SampleID`, `SampleObservationID`, `IGSN`), provenance (`Authors`,
# `DOI`, `MeasurementRun`, `Flag`, `FlagDescription`, `Comments`),
# event-level size and age attributes (`Magnitude`, `VEI`, the ages),
# deposit descriptions, and the coarser `Volcano` label. The
# exploration recorded the provenance columns as leakage and the sizes
# and ages as constants of the event.
EXCLUDED_COLUMNS = (
    "Volcano",
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
    "Magnitude",
    "VEI",
    "DepositColor",
    "DepositThickness_cm",
    "GrainSize_min",
    "GrainSize_max",
    "MeasurementRun",
    "Comments",
    "Flag",
    "FlagDescription",
)

# Deterministic duplicates of the oxide columns: `Total` is their sum
# and each `*_normalized` column is that oxide divided by `Total`. The
# `*_2SE` columns are analytical standard errors.
DERIVED_SUFFIXES = ("_normalized", "_2SE")
DERIVED_COLUMNS = ("Total", "Total_normalization")

# The 11 major oxides the table itself normalizes. The recorded
# exploration verified each one equals `oxide / Total * 100`, so this is
# the compositional block that closes to `Total`.
MAJOR_OXIDES = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeOT",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
    "Cl",
)


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the chemistry feature matrix.

    Parameters
    ----------
    frame : pandas.DataFrame
        The loaded table, including the target column.

    Returns
    -------
    pandas.DataFrame
        The feature columns only, with the target, the identifiers,
        the provenance columns and the derived duplicates removed.
    """
    features = frame.drop(
        columns=[TARGET_COL, *EXCLUDED_COLUMNS, *DERIVED_COLUMNS],
        errors="ignore",
    )
    return features.drop(
        columns=[
            column for column in features.columns if column.endswith(DERIVED_SUFFIXES)
        ]
    )


def select_major_oxides(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the major-oxide feature matrix.

    Narrower than :func:`select_features`: the trace-element block, the
    isotope ratios and their standard errors are left out, because the
    recorded exploration found trace coverage at about 20% of rows and
    absent for three volcanoes, and because whether a trace analysis was
    run follows the analytical programme rather than the geology.

    Missing oxides are kept as NaN; the learner bins them natively.

    Parameters
    ----------
    frame : pandas.DataFrame
        The loaded table, including the target column.

    Returns
    -------
    pandas.DataFrame
        The 11 major oxides, in :data:`MAJOR_OXIDES` order.
    """
    return frame[list(MAJOR_OXIDES)]
