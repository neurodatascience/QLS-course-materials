import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

from config import FIGURES_DIR, TAB10_COLORS

x, y = make_classification(
    20,
    1,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.2,
    class_sep=0.8,
    random_state=0,
)
reg = LogisticRegression().fit(x, y)

fig, ax = plt.subplots(figsize=(2.5, 2))
ax.set_yticks([0, 1])
ax.set_xticks([])
ax.scatter(x[y == 0], y[y == 0], color=TAB10_COLORS[0], marker="o")
ax.scatter(x[y == 1], y[y == 1], color=TAB10_COLORS[1], marker="^")
grid = np.linspace(*ax.get_xlim(), 100)
decision = reg.predict_proba(grid[:, None])[:, 1]
ax.plot(grid, decision, color="k")
ax.axis("off")
ax.set_title("Logistic regression")

fig.savefig(FIGURES_DIR / "logistic_regression.pdf", bbox_inches="tight")
