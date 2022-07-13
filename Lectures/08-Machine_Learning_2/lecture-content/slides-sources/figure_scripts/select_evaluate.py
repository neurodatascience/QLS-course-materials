import numpy as np
import matplotlib as mpl
from matplotlib import transforms
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
best_accuracy = accuracies[:-1][best_model_idx]
test_accuracy = accuracies[-1]
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
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.axhline(0.5, linestyle="--", color=dim_color)
ax.set_yticks([0.0, 0.5, 1.0])
ax.tick_params(axis="x", bottom=False)
sns.despine()
fig.savefig("/tmp/fig.pdf")

bbox = fig.get_tightbbox(fig.canvas.get_renderer())
x0, y0, x1, y1 = bbox.extents
fig.savefig(
    FIGURES_DIR / "select_evaluate_1.pdf",
    bbox_inches=transforms.Bbox.from_extents(x0, y0, (x0 + 2 * x1) / 3, y1),
)
xlim = ax.get_xlim()

a_x, a_y = ax.collections[0].get_offsets()[best_model_idx]
b_x, b_y = ax.collections[1].get_offsets()[0]
ax.scatter([a_x], [a_y], color=highlight_color, zorder=10)
ax.scatter([b_x], [b_y], color=highlight_color, zorder=10)
ax.text(a_x + 0.03, a_y + 0.03, f"{best_accuracy}")
ax.text(b_x + 0.03, b_y + 0.03, f"{test_accuracy}")

ax.set_xlim(*xlim)

fig.savefig(
    FIGURES_DIR / "select_evaluate_2.pdf",
    bbox_inches=transforms.Bbox.from_extents(x0, y0, (x0 + 2 * x1) / 3, y1),
)


ax.annotate(
    "", xytext=(a_x, a_y), xy=(b_x, b_y), arrowprops=dict(arrowstyle="->")
)

ax.set_xlim(*xlim)

fig.savefig(FIGURES_DIR / "select_evaluate_3.pdf", bbox_inches=bbox)
ax.set_xticklabels(["Train:\nselect params", "Test:\nevaluate"])
fig.savefig(FIGURES_DIR / "select_evaluate_4.pdf", bbox_inches="tight")
