from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# load data
data_path = Path(__file__).resolve().parents[1] / "Phenotypic_V1_0b_preprocessed1.csv"
df = pd.read_csv(data_path)

# prepare data for matplotlib
variable = "VIQ"
df = df[["DX_GROUP", variable]].dropna()
df = df[df[variable] > 0]
autism_data = df[df["DX_GROUP"] == 1][variable]
control_data = df[df["DX_GROUP"] == 2][variable]
data = [autism_data, control_data]

# create figure
fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
ax.violinplot(data)

# further customize figure
ax.set_xticks([1, 2])
ax.set_xticklabels(["Autism", "Control"])
ax.set_xlabel("Participant group")
ax.set_ylabel(variable)

# save figure
fig_path = Path(__file__).resolve().parent / "3-remake-vis-with-all-data__matplotlib.png"
fig.savefig(fig_path, bbox_inches="tight")
