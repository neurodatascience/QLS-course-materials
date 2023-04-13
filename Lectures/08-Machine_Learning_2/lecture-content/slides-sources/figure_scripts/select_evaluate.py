import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from config import FIGURES_DIR

mpl.rc("text", usetex=False)
rng = np.random.default_rng(0)
n_samples = 16 # knockout stage: 8 + 4 + 2 + 1 + 1
n_models = 30
highlight_color = "darkorange"
dim_color = "gray"
accuracies = rng.binomial(n_samples, 0.5, size=n_models + 1) / n_samples
best_model_idx = np.argmax(accuracies[:-1])
best_accuracy = accuracies[:-1][best_model_idx]
test_accuracy = accuracies[-1]
df = pd.DataFrame(
    {"Accuracy": accuracies, "dataset": "2022\nWorld Cup"}
)
df.iloc[-1, 1] = "2026\nWorld Cup"

fig, ax = plt.subplots(figsize=(2.5, 3))
jitter = True
np.random.seed(0)
sns.stripplot(
    data=df,
    x="dataset",
    y="Accuracy",
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
x_cut = (x0 + 2 * x1) / 3 - .1

xlim = ax.get_xlim()

a_x, a_y = ax.collections[0].get_offsets()[best_model_idx]
b_x, b_y = ax.collections[1].get_offsets()[0]
octopus = ax.text(a_x + 0.02, a_y + 0.03, "🐙", fontname="Symbola", fontsize=22)
monkey = ax.text(.1, 0.60, "🐒", fontname="Symbola", fontsize=22)
elephant = ax.text(.1, 0.4, "🐘", fontname="Symbola", fontsize=22)
chicken = ax.text(.0, 0.07, "🐔", fontname="Symbola", fontsize=22)

fig.savefig(
    FIGURES_DIR / "select_evaluate_1.pdf",
    bbox_inches=transforms.Bbox.from_extents(x0, y0, x_cut, y1),
)

ax.scatter([a_x], [a_y], color=highlight_color, zorder=10)
ax.scatter([b_x], [b_y], color=highlight_color, zorder=10)
ax.text(a_x - 0.38, a_y + 0.01, f"{best_accuracy:.2f}")
t_yay = ax.text(a_x + 0.29, a_y + 0.08, "– Yay!", fontsize=8)
ax.text(b_x + 0.03, b_y + 0.02, f"{test_accuracy:.2f}")

ax.set_xlim(*xlim)

fig.savefig(
    FIGURES_DIR / "select_evaluate_2.pdf",
    bbox_inches=transforms.Bbox.from_extents(x0, y0, x_cut, y1),
)

t_yay.set_visible(False)
for animal in (monkey, elephant, chicken):
    animal.set_visible(False)

t_ball = ax.text(a_x + 0.38, a_y + 0.08, "⚽", fontsize=8, font="Symbola")
t_q = ax.text(a_x + 0.29, a_y + 0.08, "–     ?", fontsize=8)

fig.savefig(
    FIGURES_DIR / "select_evaluate_2b.pdf",
    bbox_inches=transforms.Bbox.from_extents(x0, y0, x_cut, y1),
)

t_ball.set_visible(False)
t_q.set_visible(False)

ax.annotate(
    "", xytext=(a_x, a_y), xy=(b_x, b_y), arrowprops=dict(arrowstyle="->")
)

ax.text(b_x + 0.03, b_y + 0.10, "🐙", fontname="Symbola", fontsize=22)
t_oops = ax.text(b_x - 0.36, b_y + 0.14, "Oops! –", fontsize=8)
ax.set_xlim(*xlim)

fig.savefig(FIGURES_DIR / "select_evaluate_3.pdf", bbox_inches=bbox)
t_oops.set_visible(False)
fig.savefig(FIGURES_DIR / "select_evaluate_3b.pdf", bbox_inches=bbox)
ax.set_xticklabels(["Train:\nselect params", "Test:\nevaluate"])
fig.savefig(FIGURES_DIR / "select_evaluate_4.pdf", bbox_inches="tight")
