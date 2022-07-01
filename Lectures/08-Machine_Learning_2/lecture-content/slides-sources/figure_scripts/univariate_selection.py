import numpy as np
from matplotlib import pyplot as plt

from config import FIGURES_DIR

figsize = (4, 1.2)
msize = 10
rng = np.random.default_rng(0)

x1, x2, y = rng.multivariate_normal(
    [0, 0, 0], [[1.0, 0, 0.8], [0, 1, 0], [0.8, 0, 1.0]], size=20
).T
fig, axes = plt.subplots(1, 3, figsize=figsize)
axes[0].scatter(x1, y, s=msize)
axes[0].set_ylabel(r"$\y$")
axes[0].set_xlabel(r"$\X_1$")
axes[0].set_title("Keep")
axes[1].scatter(x2, y, s=msize)
axes[1].set_xlabel(r"$\X_2$")
axes[1].set_title("Drop")
axes[2].text(
    0.1,
    0.5,
    r"\huge $\dots$",
    ha="left",
    va="center",
    transform=axes[2].transAxes,
)
axes[2].axis("off")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# ax.axis("off")
fig.savefig(FIGURES_DIR / "regression.pdf", bbox_inches="tight")

plt.close("all")

y = np.zeros(20, dtype=int)
y[len(y) // 2 :] = 1

x1 = np.empty(len(y))
mean_0, mean_1 = 0, 3
x1[y == 0] = rng.normal(mean_0, size=len(y) // 2)
x1[y == 1] = rng.normal(mean_1, size=len(y) // 2)

x2 = np.empty(len(y))
mean_0, mean_1 = 0, 0.2
x2[y == 0] = rng.normal(mean_0, size=len(y) // 2)
x2[y == 1] = rng.normal(mean_1, size=len(y) // 2)

fig, axes = plt.subplots(1, 3, figsize=figsize)
axes[0].scatter(x1[y == 0], y[y == 0], s=msize)
axes[0].scatter(x1[y == 1], y[y == 1], marker="^", s=msize)
axes[1].scatter(x2[y == 0], y[y == 0], s=msize)
axes[1].scatter(x2[y == 1], y[y == 1], marker="^", s=msize)

axes[0].set_ylabel(r"$\y$")
axes[0].set_xlabel(r"$\X_1$")
axes[0].set_title("Keep")
axes[1].set_xlabel(r"$\X_2$")
axes[1].set_title("Drop")
axes[2].text(
    0.1,
    0.5,
    r"\huge $\dots$",
    ha="left",
    va="center",
    transform=axes[2].transAxes,
)
axes[2].axis("off")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])


fig.savefig(FIGURES_DIR / "classification.pdf", bbox_inches="tight")
