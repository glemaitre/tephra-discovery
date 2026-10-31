"""Data loading for the volcanic geochemistry tables.

Owns: how the raw table is read into a frame. The X/y split and the
feature selection live in `features.py`; paths are decided by the
caller, so this module never invents a `data/` directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COL = "Event"


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Read the training table from `data_dir`.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding `train.csv`.

    Returns
    -------
    pandas.DataFrame
        The training table, unmodified.
    """
    return pd.read_csv(Path(data_dir) / "train.csv")
