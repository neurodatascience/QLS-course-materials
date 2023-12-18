import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import transforms
from matplotlib import pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, TAB20_COLORS

mpl.rc("text", usetex=False)

rng = np.random.default_rng(0)
n_samples = 16
n_models = 30
n_experiments = 50

accuracies = rng.binomial(n_samples, 0.5, size=(n_experiments, n_models)) / n_samples
best_accuracies = accuracies.max(axis=1)
avg_best = best_accuracies.mean()
test_accuracies = rng.binomial(n_samples, 0.5, size=(n_experiments,)) / n_samples
df = pd.concat(
    [
        pd.DataFrame({"accuracy": best_accuracies, "dataset": "Train:\nselect predictor"}),
        pd.DataFrame({"accuracy": test_accuracies, "dataset": "Test:\nevaluate"}),
    ]
)

fig, ax = plt.subplots(figsize=(3, 3))
sns.violinplot(data=df, x="dataset", y="accuracy", color=TAB20_COLORS[2])
sns.despine()
ax.set_xlabel("")
ax.set_yticks([0.0, 0.5, 1.0])
ax.axhline(0.5, linestyle="--", color="gray")

bbox = fig.get_tightbbox(fig.canvas.get_renderer())
x0, y0, x1, y1 = bbox.extents
extents = (x0, y0, x1 + (x1 - x0) * 0.1, y1)
fig.savefig(
    FIGURES_DIR / "select_evaluate_averaged_1.pdf",
    bbox_inches=transforms.Bbox.from_extents(*extents),
)
ax.axhline(avg_best, linestyle="--", color="gray")
x = 1.3
ax.annotate(
    "",
    xytext=(x, 0.5),
    xy=(x, avg_best),
    arrowprops=dict(arrowstyle="<->"),
)
ax.text(x + 0.05, 0.6, "Bias")

fig.savefig(
    FIGURES_DIR / "select_evaluate_averaged_2.pdf",
    bbox_inches=transforms.Bbox.from_extents(*extents),
)
