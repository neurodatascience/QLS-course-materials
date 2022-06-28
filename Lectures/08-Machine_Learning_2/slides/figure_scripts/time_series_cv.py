import numpy as np
from matplotlib import pyplot as plt

from config import FIGURES_DIR

def show_cv(shuffle):
    x = np.linspace(0, 3 * np.pi, 50)
    rng = np.random.default_rng(0)
    y = np.sin(x) + rng.normal(size=len(x)) * 0.3
    if shuffle:
        order = np.arange(len(x))
        rng.shuffle(order)
        x, y = x[order], y[order]

    fig, ax = plt.subplots(figsize=(3, 2))
    n_train = int(0.7 * len(x))
    ax.scatter(x[:n_train], y[:n_train])
    ax.scatter(x[n_train:], y[n_train:], marker="^")
    ax.legend(
        ["Train", "Test"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

fig = show_cv(False)
fig.savefig(FIGURES_DIR / "kfold.pdf", bbox_inches="tight")

fig = show_cv(True)
fig.savefig(FIGURES_DIR / "kfold_shuffled.pdf", bbox_inches="tight")
