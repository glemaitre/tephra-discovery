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
