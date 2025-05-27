from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load data
data_path = Path(__file__).resolve().parents[3] / "data" / "participants_nbsub-200.tsv"
df = pd.read_csv(data_path, sep="\t")

# prepare data for seaborn
variable = "VIQ"
df = df[["DX_GROUP", variable]].dropna()
df = df[df[variable] > 0]

# create the Figure and Axes artists
fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")

# plot the
sns.swarmplot(df, x="DX_GROUP", y=variable, ax=ax)

# further customize figure
ax.set_xticklabels(["Autism", "Control"])

# save figure
fig_path = Path(__file__).resolve().parent / "3-remake-vis-with-all-data__seaborn.png"
fig.savefig(fig_path, bbox_inches="tight")
