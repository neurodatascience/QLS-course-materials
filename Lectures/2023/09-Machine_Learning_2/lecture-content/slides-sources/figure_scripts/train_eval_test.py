from config import FIGURES_DIR, TAB10_COLORS
from matplotlib import pyplot as plt

COLORS = [list(c[:3]) + [0.7] for c in TAB10_COLORS]

fig, axes = plt.subplots(1, 3, figsize=(5, 1.2), gridspec_kw={"wspace": 0})

titles = ["Train", "Validation", "Test"]
descriptions = [
    r"""Choose $\beta(\alpha_i)$
for each $\alpha_i$""",
    r"""Evaluate $\beta(\alpha_i)$;
Choose $\alpha$""",
    r"Evaluate $\beta(\alpha)$",
]
for ax, c, d, t in zip(axes, COLORS, descriptions, titles):
    ax.set_facecolor(c)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, d, ha="center", va="center")
    ax.set_title(t)

fig.savefig(FIGURES_DIR / "datasets.pdf", bbox_inches="tight")

plt.close("all")


def add_cv(fig, cv_gs, test_name="Valid."):
    for inner_split in range(3):
        for fold in range(3):
            ax = fig.add_subplot(cv_gs[inner_split, fold])
            if fold == 2 - inner_split:
                ax.set_facecolor(COLORS[1])
                ax.text(0.5, 0.5, test_name, ha="center", va="center")
            else:
                ax.set_facecolor(COLORS[0])
                ax.text(0.5, 0.5, "Train", ha="center", va="center")


fig = plt.figure(figsize=(6, 4))
outer_gs = fig.add_gridspec(2, 1, hspace=0.4)
for outer_split in range(2):
    inner_gs = outer_gs[outer_split].subgridspec(2, 2, height_ratios=[3, 1], wspace=0.05)
    test_idx = 1 - outer_split
    refit_ax = fig.add_subplot(inner_gs[-1, outer_split])
    refit_ax.text(0.5, 0.5, "Refit", ha="center", va="center")
    refit_ax.set_facecolor(COLORS[0])
    test_ax = fig.add_subplot(inner_gs[-1, test_idx])
    test_ax.text(0.5, 0.5, "Test", ha="center", va="center")
    test_ax.set_facecolor(COLORS[2])
    cv_gs = inner_gs[0, outer_split].subgridspec(3, 3)
    add_cv(fig, cv_gs)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(FIGURES_DIR / "cv.pdf", bbox_inches="tight")

plt.close("all")

fig = plt.figure(figsize=(3, 1.5))
gs = fig.add_gridspec(3, 3, wspace=0.1)
add_cv(fig, gs, "Test")
for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(FIGURES_DIR / "cv_not_nested.pdf", bbox_inches="tight")
