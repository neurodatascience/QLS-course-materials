"""Generate dataset by extracting connectivity between brain regions.

Works on the ABIDE dataset.

TODO:

-   Add listing of regions used
"""

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

output_dir: Path = Path(__file__).parent

n_subjects: int = 200

# data specific to regions defined by
# the Automatic Anatomical Labeling atlas.
parcellation = "rois_aal"
# the Harvard-Oxford atlas.
parcellation: str = "rois_ho"

# connectivity measure
kind: str = "correlation"


def main() -> None:
    data = datasets.fetch_abide_pcp(n_subjects=n_subjects, derivatives=[parcellation])

    pheno = pd.DataFrame(data["phenotypic"]).drop(columns=["i", "Unnamed: 0"])
    pheno = pheno.fillna("n/a")
    pheno.to_csv(output_dir / f"participants_nbsub-{n_subjects}.tsv", sep="\t", index=False)

    features = data[parcellation]

    # Here we are pre-processing each image independently
    # that is not using any group-level information for scaling / normalization / feature transformation (e.g. PCA).
    # We can do on entire dataset without creating train-test splits first
    # and then defining feature-set on the training data only.
    flat_features_list = []
    for subject in features:
        flat_features = extract_connectome_features(subject, measure=ConnectivityMeasure(kind=kind))
        flat_features_list.append(flat_features)

    X = np.array(flat_features_list)

    df = pd.DataFrame(X, index=pheno["SUB_ID"])

    df.to_csv(
        output_dir
        / f"abide_nbsub-{n_subjects}_atlas-{parcellation.split('_')[1]}_meas-{kind}_relmat.tsv",
        sep="\t",
    )


def extract_connectome_features(func_data, measure=None):
    """Calculate connnectome based on timeseries data and similarity measure."""
    if measure is None:
        measure = ConnectivityMeasure(kind="correlation")

    connectome_matrix = measure.fit_transform([func_data])[0]
    print(connectome_matrix.shape)

    # Extract lower (or upper) triangle entrees (excluding diagonal)
    tril_idx = np.tril_indices(len(connectome_matrix), k=-1)
    flat_features = connectome_matrix[tril_idx]

    return flat_features


if __name__ == "__main__":
    main()
