import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from config import FIGURES_DIR

mpl.rc("text", usetex=False)
rng = np.random.default_rng(0)
n_samples = 20
n_models = 30
highlight_color = "darkorange"
dim_color = "gray"
accuracies = rng.binomial(n_samples, 0.5, size=n_models + 1) / n_samples
best_model_idx = np.argmax(accuracies[:-1])
df = pd.DataFrame({"accuracy": accuracies, "dataset": "Dataset A"})
df.iloc[-1, 1] = "Dataset B"

fig, ax = plt.subplots(figsize=(2.5, 3))
jitter = True
np.random.seed(0)
sns.stripplot(
    data=df,
    x="dataset",
    y="accuracy",
    color=dim_color,
    alpha=0.5,
    jitter=jitter,
)
a_x, a_y = ax.collections[0].get_offsets()[best_model_idx]
b_x, b_y = ax.collections[1].get_offsets()[0]
ax.scatter([a_x], [a_y], color=highlight_color, zorder=10)
ax.scatter([b_x], [b_y], color=highlight_color, zorder=10)
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.annotate(
    "", xytext=(a_x, a_y), xy=(b_x, b_y), arrowprops=dict(arrowstyle="->")
)
ax.axhline(0.5, linestyle="--", color=dim_color)
# ax.set_yticks(list(ax.get_yticks()) + [0.5])
ax.set_yticks([0.0, 0.5, 1.0])
ax.tick_params(axis="x", bottom=False)
sns.despine()

fig.savefig(FIGURES_DIR / "select_evaluate.pdf", bbox_inches="tight")
