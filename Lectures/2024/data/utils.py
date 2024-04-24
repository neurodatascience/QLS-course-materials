"""Utilities"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[1] / "data"


def data_loader(n_subjects: int = 100, parcellation: str = "rois_ho"):
    """Load connectictiy data from the ABIDE dataset.

    Parameters
    ----------
    n_subjects : number of subjects to load

    parcellation : parcellation to use
                   can be  ``"rois_ho"`` or ``"rois_aal"``

    To use it:

    .. code-block:: python

        import sys
        sys.path.insert(0, "/home/remi/github/origami/QLS-course-materials/Lectures/2024/")

        from data.utils import data_loader

        data, participants = data_loader()
    """
    if parcellation not in ["rois_ho", "rois_aal"]:
        raise ValueError(
            "'parcellation' must be one of ['rois_ho, 'rois_aal'].\n" f"Got: {parcellation}"
        )

    input_file = (
        DATA_DIR
        / f"abide_nbsub-{n_subjects}_atlas-{parcellation.split('_')[1]}_meas-correlation_relmat.tsv"
    )

    return pd.read_csv(input_file, sep="\t"), pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
