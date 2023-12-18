from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def stevens_law(intensity, N):
    sensation = intensity**N
    return sensation


# define data
data = {
    "Electric Shock": 3.5,
    "Saturation": 1.7,
    "Length": 1,
    "Area": 0.7,
    "Depth": 0.67,
    "Brightness": 0.5,
}

colors = ["lightblue", "lightgreen", "purple", "red", "darkgreen", "gray"]

text_locs = [(1.6, 4.2), (2.5, 4.1), (3.9, 3.6), (3.4, 2.7), (3.7, 2.0), (3.5, 1.3)]

rotations = [85, 70, 45, 30, 20, 10]

x = np.arange(0, 5.1, 0.1)
ys = []
for label, N in data.items():
    y = stevens_law(x, N)
    ys.append(y)

# create figure
fig, ax = plt.subplots(1, 1, figsize=[3, 3])
for y, color, (stimuli, N), (x_text, y_text), rotation in zip(
    ys, colors, data.items(), text_locs, rotations
):
    ax.plot(
        x,
        y,
        color=color,
    )
    label = f"{stimuli} ({N})"
    ax.text(x=x_text, y=y_text, s=label, rotation=rotation, color=color)

ax.set_ylim([0, 6])
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("Perceived Sensation")
ax.set_xlabel("Physical Intensity")
ax.set_title("Steven's Psychophysical Power Law: $S=I^N$\n\n\n")

# save figure
fig_path = Path(__file__).resolve().parent / "line_graph_stevens_law_imitation.png"
fig.savefig(fig_path, bbox_inches="tight")
