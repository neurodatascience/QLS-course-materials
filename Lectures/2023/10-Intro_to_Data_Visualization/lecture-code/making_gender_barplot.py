from pathlib import Path

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(3, 3))

x = ["M", "F"]
y = [396, 51]
color = ["#007bb8ff", "#ffc6d0ff"]

ax.bar(x=x, height=y, color=color, edgecolor="k")

ax.set_ylabel("N    ", rotation=0)


fig_path = Path(__file__).resolve().parent / "making_gender_barplot.svg"
fig.savefig(fig_path, bbox_inches="tight")
