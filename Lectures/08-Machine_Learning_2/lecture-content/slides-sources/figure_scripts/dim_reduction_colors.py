import numpy as np
from scipy.linalg import svd, lstsq
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
from sklearn.decomposition import NMF

from config import FIGURES_DIR


def paint(axes, colors):
    for col, ax in zip(colors, axes):
        ax.set_facecolor(col)


def show_colors(X, W, y):
    n_colors = X.shape[1]
    fig, axes = plt.subplots(3, n_colors, figsize=(6, 4))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    paint(axes[0], X.T)
    paint(axes[1, 2:], W.T)
    paint(axes[2, 3:], [y])
    return fig, axes


def make_data(n_components=2, seed=5):
    colors = mcolors.TABLEAU_COLORS
    rng = np.random.default_rng(seed)
    color_sample = rng.choice(list(colors.values()), size=8, replace=False)
    color_sample = np.asarray([mcolors.to_rgb(c) for c in color_sample]).T
    X = color_sample[:, :-1]
    y = color_sample[:, -1]
    init = "nndsvda" if n_components < min(X.shape) else "random"
    nmf = NMF(n_components, init=init, random_state=seed)
    W = nmf.fit_transform(X)
    # W_scaled = W / np.linalg.norm(W, axis=0)
    W_scaled = W / W.max(axis=0)
    proj = nmf.inverse_transform(W)
    proj_scaled = proj / proj.max(axis=0)
    # proj_clipped = np.minimum(proj, 1.)
    proj_clipped = proj / np.maximum(1, proj.max(axis=0))
    full_coef, *_ = lstsq(X, y)
    reduced_coef, *_ = lstsq(W, y)
    y_hat = W.dot(reduced_coef)
    return {
        "X": X,
        "y": y,
        "full_coef": full_coef,
        "y_hat_reduced": y_hat,
        "reduced_coef": reduced_coef,
        "W": W,
        "W_scaled": W_scaled,
        "H": nmf.components_,
        "proj": proj,
        "proj_scaled": proj_scaled,
        "proj_clipped": proj_clipped,
    }


def paint_axes(ax, colors, top_right_align=False):
    ax.set_axis_off()
    w = 1 / len(colors)
    start = .2 * w if top_right_align else 0.
    y0 = .1 if top_right_align else .05
    for c in colors:
        rect = mpatches.Rectangle(
            (start, y0),
            w * 0.8,
            0.9,
            transform=ax.transAxes,
            edgecolor="k",
            facecolor=c,
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


def show_regression(data, reduced_dim, show_beta_horizontal_barplot=False, w_name=None):
    # figw = 2.5 if reduced_dim else 3.5
    figw = 2.5
    fig = plt.figure(figsize=(figw, 3))
    X = data["W_scaled"].T if reduced_dim else data["X"].T
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, len(X), 0.0, 1],
        height_ratios=[1.0, 0.25],
        hspace=0.,
    )
    y_ax = fig.add_subplot(gs[0, 0])
    y_ax.set_title(r"$\y$")
    paint_axes(y_ax, [data["y"]])
    y_hat_ax = fig.add_subplot(gs[0, 2])
    y_hat_ax.set_title(r"$\hat{\y}$")
    paint_axes(y_hat_ax, [data["y_hat_reduced"] if reduced_dim else data["y"]])
    x_ax = fig.add_subplot(gs[0, 4])
    if w_name is None:
        w_name = r"$\W$" if reduced_dim else r"$\X$"
    x_ax.set_title(w_name)
    paint_axes(x_ax, X)
    approx_ax = fig.add_subplot(gs[0, 1])
    write_axes(approx_ax, r"$\approx$")
    equal_ax = fig.add_subplot(gs[0, 3])
    write_axes(equal_ax, "=")
    if show_beta_horizontal_barplot:
        beta_h_ax = fig.add_subplot(gs[0, 6])
        beta_h_ax.set_title(r"$\hat{\bbeta}$")
        show_vector(
            beta_h_ax,
            data["reduced_coef"] if reduced_dim else data["full_coef"],
            "h",
        )
    beta_v_ax = fig.add_subplot(gs[1, 4])
    show_vector(
        beta_v_ax,
        data["reduced_coef"] if reduced_dim else data["full_coef"],
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


def show_factorization(data):
    fig = plt.figure(figsize=(6, 3))
    n_dim = data["X"].shape[1]
    n_components = data["W"].shape[1]
    gs = fig.add_gridspec(
        3,
        7,
        width_ratios=[n_dim, 2, n_dim, 2, n_components, 0.0, n_dim],
        height_ratios=[n_components, 1.5 * n_dim, 0],
        hspace=0,
        wspace=0
    )
    x_ax = fig.add_subplot(gs[1, 0])
    x_ax.set_title(r"$\X$")
    paint_axes(x_ax, data["X"].T, True)
    proj_ax = fig.add_subplot(gs[1, 2])
    # proj_ax.set_title(r"$\X'$")
    proj_ax.set_title(r"$\W \, \bH$")
    # paint_axes(proj_ax, data["proj_scaled"].T)
    paint_axes(proj_ax, data["proj_clipped"].T, True)
    w_ax = fig.add_subplot(gs[1, 4])
    w_ax.set_title(r"$\W$")
    paint_axes(w_ax, data["W_scaled"].T, True)
    approx_ax = fig.add_subplot(gs[1, 1])
    write_axes(approx_ax, r"$\approx$")
    equal_ax = fig.add_subplot(gs[1, 3])
    write_axes(equal_ax, "=")
    H_ax = fig.add_subplot(gs[0, 6])
    H_ax.imshow(data["H"], cmap="Greys")
    H_ax.set_title(r"$\bH$")
    # H_ax.set_axis_off()
    H_ax.set_xticks([])
    H_ax.set_yticks([])
    return fig


for n_components in [1, 2, 3, 4, 7]:
    print(f"{n_components} components")
    data = make_data(n_components, 0)
    fig = show_regression(data, False)
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGURES_DIR / f"regression_full_{n_components}.{ext}",
            bbox_inches="tight",
        )
    plt.close("all")
    fig = show_regression(data, True)
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGURES_DIR / f"regression_reduced_{n_components}.{ext}",
            bbox_inches="tight",
        )
    plt.close("all")
    if n_components == 3:
        fig = show_regression(data, True, w_name=r"$\mathbold{U}$")
        for ext in ("pdf", "png"):
            fig.savefig(
                FIGURES_DIR / f"regression_reduced_{n_components}_svd.{ext}",
                bbox_inches="tight",
            )
        plt.close("all")
    fig = show_factorization(data)
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGURES_DIR / f"factorization_{n_components}.{ext}",
            bbox_inches="tight",
        )
    plt.close("all")
