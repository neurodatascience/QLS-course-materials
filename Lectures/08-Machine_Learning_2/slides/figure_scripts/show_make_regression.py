from matplotlib import pyplot as plt

from config import FIGURES_DIR, TAB10_COLORS
COLORS = [list(c[:3]) + [0.7] for c in TAB10_COLORS]

fig, axes = plt.subplots(
    1, 4, figsize=(3, 2), gridspec_kw=dict(width_ratios=[1.0, 1.0, 5.0, 7.0])
)

axes[0].text(0.5, 0.5, r"$\X$", ha="center", va="center")
axes[0].axis("off")
axes[1].text(0.5, 0.5, r"$=$", ha="center", va="center")
axes[1].axis("off")
axes[2].text(
    0.5,
    0.5,
    r"""Useful
features""",
    ha="center",
    va="center",
)
axes[2].set_facecolor(COLORS[0])
axes[3].text(0.5, 0.5, r"Noise", ha="center", va="center")
axes[3].set_facecolor(COLORS[1])

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(FIGURES_DIR / "x_construction.pdf", bbox_inches="tight")
