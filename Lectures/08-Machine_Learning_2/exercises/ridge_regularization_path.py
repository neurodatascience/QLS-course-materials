import numpy as np
from sklearn.linear_model import Ridge
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl

mpl.rc("text", usetex=True)
mpl.rc(r"\usepackage{DejaVuSans} \usepackage{eulervm}")

try:
    rng = np.random.default_rng(0)
except AttributeError:
    rng = np.random.RandomState(0)
coef = np.zeros(50)
coef[:3] = [10, -10, 20]
alpha_grid = np.logspace(-2, 3, 50)
n_simu = 100
all_estimated_coef = []
all_alpha = []
all_scores = []
all_train_scores = []
noise = 10
for i in range(n_simu):
    X = rng.normal(size=(80, len(coef)))
    y = X.dot(coef) + rng.normal(size=len(X)) * noise
    n_train = len(X) // 2
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    for j, alpha in enumerate(alpha_grid):
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        all_estimated_coef.append(model.coef_)
        all_alpha.append(alpha)
        all_scores.append(model.score(X_test, y_test))
        all_train_scores.append(model.score(X_train, y_train))


df = pd.DataFrame(all_estimated_coef)
df["alpha"] = all_alpha
df["score"] = all_scores
df["train_score"] = all_train_scores
fig, axes = plt.subplots(
    5,
    1,
    sharex=True,
    sharey=False,
    figsize=(4, 8),
    gridspec_kw=dict(height_ratios=[2, 1, 1, 1, 1]),
)
for c, (ax, true_coef) in enumerate(zip(axes[1:], coef)):
    sns.lineplot(
        data=df,
        x="alpha",
        y=c,
        ci="sd",
        ax=ax,
        color=list(colors.TABLEAU_COLORS.values())[2],
    )
    ax.axhline(true_coef, linestyle="--", color="k")
    ax.axhline(0, linestyle="-", color="k")
    y0, y1 = ax.get_ylim()
    d = (y1 - y0) * 0.1
    ax.set_ylim(y0 - d, y1 + d)
    ax.set_title(rf"$\beta_{c + 1}$")
    ax.set_ylabel("")
    # sns.lineplot(data=df, x="alpha", y=c, ax=ax)
    # ax.set_xscale("log")
sns.lineplot(data=df, x="alpha", y="score", ax=axes[0], ci="sd")
sns.lineplot(data=df, x="alpha", y="train_score", ax=axes[0], ci="sd")
axes[0].legend([r"$R^2$ on test set", r"$R^2$ on train set"], frameon=False)
axes[0].set_title(r"$R^2$ score")
axes[0].set_ylabel("")
axes[0].set_xscale("log")
# axes[0].set_ylim(-.1, 1.1)
axes[-1].set_xlabel(r"$\lambda$")
# axes[1].legend([r"$\hat{\beta}$", r"$\beta$"], frameon=False)
axes[1].legend(["estimate", "true coef"], frameon=False, loc="center right")
axes[-1].text(
    0,
    -0.35,
    r"$\leftarrow$ less regularized",
    transform=axes[-1].transAxes,
    ha="left",
)
axes[-1].text(
    1,
    -0.35,
    r"more regularized $\rightarrow$",
    transform=axes[-1].transAxes,
    ha="right",
)
plt.tight_layout()
fig.savefig(
    "ridge_regularization_path.pdf", bbox_inches="tight", transparent=True
)
