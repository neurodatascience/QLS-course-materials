from pathlib import Path

import matplotlib.pyplot as plt

# data for the Verbal IQ summary statistics
groups = ["Autism", "Control"]
means = [105, 112]
stdevs = [17.4, 13.3]

# create the Figure and Axes artists
fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")

# plot the summary statistics in a bar graph with error bars
ax.bar(x=groups, height=means, yerr=stdevs, capsize=10)

# further customize the figure
ax.set_ylabel("Verbal IQ")


# save the figure
fig_path = Path(__file__).resolve().parent / "2-encode-a-simple-vis.png"
fig.savefig(fname=fig_path, bbox_inches="tight")
