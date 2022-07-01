from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.stats

from config import FIGURES_DIR

rng = np.random.default_rng(0)
n = 20
x = rng.normal(size=n)
e = rng.normal(size=n) / 5
y = 0.5 * x + 0.1 * x ** 2 + e
reg = scipy.stats.linregress(x, y)

fig, ax = plt.subplots(figsize=(2.5, 2))
ax.scatter(x, y)
grid = np.sort(x)
ax.plot(grid, reg.intercept + reg.slope * grid, color="k")
ax.axis("off")
ax.set_title("Linear regression")
fig.savefig(FIGURES_DIR / "linear_regression.pdf", bbox_inches="tight")
