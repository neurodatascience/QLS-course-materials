from pathlib import Path

import pandas as pd
from nilearn import datasets
from rich import print

output_dir = Path(__file__).parent

n_subjects = 100

# data specific to regions defined by

# the Automatic Anatomical Labeling atlas.
parcel = "rois_aal"
# the Harvard-Oxford atlas.
parcel = "rois_ho"

data = datasets.fetch_abide_pcp(derivatives=[parcel])
print(data)

pheno = pd.DataFrame(data["phenotypic"]).drop(columns=["i", "Unnamed: 0"])
pheno.to_csv(output_dir / "participants.tsv")
