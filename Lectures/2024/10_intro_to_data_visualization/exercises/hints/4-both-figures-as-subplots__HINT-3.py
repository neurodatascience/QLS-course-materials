from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# summary stats data for Verbal IQ
groups = ["Autism", "Control"]
means = [105, 112]
stdevs = [17.4, 13.3]

# full data for Verbal IQ
data_path = Path(__file__).resolve().parents[3] / "data" / "participants_nbsub-200.tsv"
df = pd.read_csv(data_path, sep="\t")

variable = "VIQ"
df = df[["DX_GROUP", variable]]
df = df.dropna()
df = df[df[variable] > 0]

# create the Figure and Axes artists, with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 5), layout="constrained")

# plot the summary stats on the first axis
axs[0].bar(x=groups, height=means, yerr=stdevs, capsize=10)

# plot the full data on the second axis
sns.swarmplot(df, x="DX_GROUP", y=variable, ax=axs[1])

# further customize the figures
ylim_lower = min(df[variable]) - 20
ylim_upper = max(df[variable]) + 20
for ax in axs:
    ax.set_ylim([ylim_lower, ylim_upper])

axs[1].set_xticklabels(groups)
axs[0].set_ylabel("Verbal IQ")
axs[1].set_ylabel("")

fig_path = Path(__file__).resolve().parent / "4-both-figures-as-subplots.png"
fig.savefig(fig_path, bbox_inches="tight")
