from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load data
data_path = Path(__file__).resolve().parents[1] / "Phenotypic_V1_0b_preprocessed1.csv"
df = pd.read_csv(data_path)

# prepare data for seaborn
variable = "VIQ"
df = df[["DX_GROUP", variable]].dropna()
df = df[df[variable] > 0]

# create the Figure and Axes artists
fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")

# plot the
sns.violinplot(df, x="DX_GROUP", y=variable, ax=ax)

# further customize figure
ax.set_xticklabels(["Autism", "Control"])

# save figure
fig_path = Path(__file__).resolve().parent / "3-remake-vis-with-all-data__seaborn.png"
fig.savefig(fig_path, bbox_inches="tight")
