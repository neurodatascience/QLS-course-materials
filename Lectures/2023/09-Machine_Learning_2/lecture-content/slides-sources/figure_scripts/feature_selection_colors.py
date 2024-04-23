import matplotlib.patches as mpatches
import numpy as np
from config import FIGURES_DIR
from matplotlib import colors as mcolors, pyplot as plt
from scipy import linalg
from sklearn import feature_selection


def paint(axes, colors):
    for col, ax in zip(colors, axes):
        ax.set_facecolor(col)


def paint_axes(ax, colors, top_right_align=False, linestyles=None):
    ax.set_axis_off()
    w = 1 / len(colors)
    start = 0.2 * w if top_right_align else 0.0
    y0 = 0.1 if top_right_align else 0.05
    for i, c in enumerate(colors):
        kwarg = {"linestyle": linestyles[i]} if linestyles is not None else {}
        rect = mpatches.Rectangle(
            (start, y0),
            w * 0.8,
            0.9,
            transform=ax.transAxes,
            edgecolor="k",
            facecolor=c,
            **kwarg,
        )
        ax.add_artist(rect)
        start += w


def write_axes(ax, text):
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        rf"\large{{{text}}}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


def show_vector(ax, coords, orient, positive=False):
    assert orient in ("h", "v")
    ax.set_axis_off()
    if positive:
        coords = coords - coords.min()
    color = "gray"
    if orient == "h":
        ax.barh(range(len(coords)), coords, color=color)
        ax.set_ylim(9, -1)
    else:
        ax.bar(range(len(coords)), coords, color=color)


def show_regression(data, full_x=False, show_beta_horizontal_barplot=False):
    # figw = 2.5 if reduced_dim else 3.5
    figw = 2.5
    fig = plt.figure(figsize=(figw, 3))
    X = data["X"].T if full_x else data["X_transformed"].T
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, len(X), 0.0, 1],
        height_ratios=[1.0, 0.25],
        hspace=0.0,
    )
    y_ax = fig.add_subplot(gs[0, 0])
    y_ax.set_title(r"$\y$")
    paint_axes(y_ax, [data["y"]])
    y_hat_ax = fig.add_subplot(gs[0, 2])
    y_hat_ax.set_title(r"$\hat{\y}$")
    paint_axes(y_hat_ax, [data["y_hat"]])
    x_ax = fig.add_subplot(gs[0, 4])
    x_ax.set_title(r"$\X$")
    if full_x:
        ls = ["--" for i in range(X.shape[0])]
        for idx in data["kept_idx"]:
            ls[idx] = "-"
        kwarg = {"linestyles": ls}
    else:
        kwarg = {}
    paint_axes(x_ax, X, **kwarg)
    approx_ax = fig.add_subplot(gs[0, 1])
    write_axes(approx_ax, r"$\approx$")
    equal_ax = fig.add_subplot(gs[0, 3])
    write_axes(equal_ax, "=")
    if show_beta_horizontal_barplot:
        beta_h_ax = fig.add_subplot(gs[0, 6])
        beta_h_ax.set_title(r"$\hat{\bbeta}$")
        show_vector(
            beta_h_ax,
            data["coef_full"] if full_x else data["coef"],
            "h",
        )
    beta_v_ax = fig.add_subplot(gs[1, 4])
    show_vector(
        beta_v_ax,
        data["coef_full"] if full_x else data["coef"],
        "v",
    )
    beta_v_ax.text(
        0.5,
        -0.1,
        r"\large $\hat{\bbeta}$",
        ha="center",
        va="top",
        transform=beta_v_ax.transAxes,
    )
    return fig


def make_data(seed=0):
    colors = mcolors.TABLEAU_COLORS
    rng = np.random.default_rng(seed)
    color_sample = rng.choice(list(colors.values()), size=8, replace=False)
    color_sample = np.asarray([mcolors.to_rgb(c) for c in color_sample]).T
    X = color_sample[:, :-1]
    y = color_sample[:, -1]
    selector = feature_selection.SelectKBest(feature_selection.f_regression, k=3)
    X_transformed = selector.fit_transform(X, y)
    coef, *_ = linalg.lstsq(X_transformed, y)
    y_hat = X_transformed.dot(coef)
    X[:, np.logical_not(selector.get_support())] = np.asarray(mcolors.to_rgb("white"))[:, None]
    coef_full = np.zeros(X.shape[1])
    coef_full[selector.get_support()] = coef
    return {
        "X_transformed": X_transformed,
        "X": X,
        "y": y,
        "coef": coef,
        "coef_full": coef_full,
        "y_hat": y_hat,
        "original_n_features": 7,
        "kept_idx": selector.get_support(True),
    }


data = make_data()
fig = show_regression(data, True)
fig.savefig(
    FIGURES_DIR / "regression_selected_3_full_coef.pdf",
    bbox_inches="tight",
)
plt.close("all")
fig = show_regression(data, False)
fig.savefig(
    FIGURES_DIR / "regression_selected_3.pdf",
    bbox_inches="tight",
)
