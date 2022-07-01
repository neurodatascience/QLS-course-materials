import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd

from config import FIGURES_DIR, TAB10_COLORS

FIGSIZE = 2, 2


def show_cloud(X, color, ax):
    ax.scatter(X[:, 0], X[:, 1], alpha=0.2, s=12, color=color)
    ax.set_aspect(1)
    ax.set_xlim(-30, 35)
    ax.set_ylim(-15, 20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$\X_1$")
    ax.set_ylabel("$\X_2$")


rng = np.random.default_rng(0)
X = rng.normal(size=(70, 2))
X = X @ np.diag((10, 2))
fig, ax = plt.subplots(figsize=FIGSIZE)
show_cloud(X, TAB10_COLORS[0], ax)
fig.savefig(str(FIGURES_DIR / "cloud_aligned.pdf"), bbox_inches="tight")
plt.close("all")

fig, ax = plt.subplots(figsize=FIGSIZE)
a = np.pi / 6
X = X @ [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]
X_mean = [15, 5]
X += X_mean
U, S, Vh = svd(X - X.mean(axis=0))
show_cloud(X, TAB10_COLORS[1], ax)
v_len = 20
ax.plot(
    *(np.asarray([X_mean - Vh[0] * v_len, X_mean + Vh[0] * v_len]).T),
    color="k"
)

fig.savefig(str(FIGURES_DIR / "cloud_not_aligned.pdf"), bbox_inches="tight")
plt.close("all")

fig, ax = plt.subplots(figsize=FIGSIZE)
show_cloud(X, "gray", ax)
v_len = 10
ax.plot(
    *(np.asarray([X_mean, X_mean + Vh[0] * v_len]).T), color=TAB10_COLORS[0]
)
ax.plot(
    *(np.asarray([X_mean, X_mean + Vh[1] * v_len]).T), color=TAB10_COLORS[1]
)
ax.legend([r"$\V_1$", r"$\V_2$"], frameon=False, handlelength=1.0)

fig.savefig(
    str(FIGURES_DIR / "cloud_not_aligned_with_pc.pdf"), bbox_inches="tight"
)
