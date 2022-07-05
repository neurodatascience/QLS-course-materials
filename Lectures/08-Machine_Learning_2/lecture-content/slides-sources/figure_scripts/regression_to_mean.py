import numpy as np
from matplotlib import pyplot as plt

from config import FIGURES_DIR


rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=(10, 30))
best = np.argmax(rolls, axis=0)

fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(rolls.T, color="gray", alpha=0.1)
(max_line,) = ax.plot(np.max(rolls, axis=0), label="Best die")
x = np.arange(1, rolls.shape[1])
(prev_max_line,) = ax.plot(
    x, rolls[best[:-1], x], label="Best die from previous roll"
)
(expected,) = ax.plot(
    3.5 * np.ones(rolls.shape[1]),
    linestyle="--",
    color="k",
    label="Expected value",
)
ax.legend(
    handles=[max_line, prev_max_line, expected],
    bbox_to_anchor=(1.0, 0.0),
    loc="upper right",
    frameon=False
)
ax.set_xticks([])
ax.set_yticks(range(1, 7))

fig.savefig(FIGURES_DIR / "dice_rolls.pdf", bbox_inches="tight")
