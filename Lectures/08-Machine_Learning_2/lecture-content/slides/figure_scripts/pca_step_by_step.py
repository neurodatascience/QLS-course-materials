import numpy as np
from matplotlib import pyplot as plt
from skimage import data, filters
from scipy.linalg import svd
from sklearn.metrics import explained_variance_score

from config import FIGURES_DIR

# img = mpl.image.imread("/tmp/astronaut.jpg")

img = data.coins()[:, :100]
img = data.brick()[:60, :20]
img = data.cat().sum(axis=-1)
# img = filters.gaussian(img, 3)[55:140:2, 150:200:2]
# img = filters.gaussian(img, 7)[:, 150:350]
img = data.binary_blobs(20, seed=0)[:, :8]
img = filters.gaussian(img, 1)
img -= img.mean(axis=0)

U, S, Vt = svd(img)
k = 3

cmap = "gray"
# axes[1].imshow(approx, cmap=cmap)


def show_decomposition(img, U, S, Vt, k):
    max_k = 3
    approx = U[:, :k].dot(np.diag(S[:k])).dot(Vt[:k, :])
    fig = plt.figure(figsize=(6, 1.8))
    h_ratios = [0.1, 2.0]
    w_ratios = [1.0, 0.5, 1.0, 0.5, 0.5] + [0.12, 0.1, 1.0, 0.8] * max_k
    gs = fig.add_gridspec(
        len(h_ratios),
        len(w_ratios),
        width_ratios=w_ratios,
        height_ratios=h_ratios,
        wspace=0.1,
        hspace=0.05,
    )
    img_ax = fig.add_subplot(gs[1, 0])
    img_ax.imshow(img, cmap=cmap)
    img_ax.text(
        0.5,
        1.01,
        r"$\X$",
        ha="center",
        va="bottom",
        transform=img_ax.transAxes,
    )
    img_ax.text(
        -0.1, 0.5, r"$n$", ha="right", va="center", transform=img_ax.transAxes
    )
    img_ax.text(
        0.5, -0.0, r"$p$", ha="center", va="top", transform=img_ax.transAxes
    )
    eq_ax = fig.add_subplot(gs[1, 1])
    eq_ax.text(0.5, 0.5, r"$\approx$", ha="center")
    explained_var = explained_variance_score(img, approx)
    eq_ax.text(
        0.5,
        -0.1,
        rf"Explained variance: {explained_var:.2f}",
        ha="center",
        va="top",
        transform=eq_ax.transAxes,
    )
    eq_ax.set_axis_off()
    approx_ax = fig.add_subplot(gs[1, 2])
    approx_ax.imshow(approx, cmap=cmap)
    approx_ax.text(
        0.5,
        1.01,
        # 1.05,
        # r"$\hat{\X}$",
        r"",
        # r"$\sum_{i=1}^k U_i\,s_i\,V_i^T$",
        ha="center",
        va="bottom",
        transform=approx_ax.transAxes,
    )
    eq_ax = fig.add_subplot(gs[1, 3])
    eq_ax.text(0.5, 0.5, r"$=$", ha="center")
    eq_ax.set_axis_off()
    show_svd_terms(fig, gs, 4, k)
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def show_svd_terms(fig, gs, start_pos, k):
    pos = start_pos
    for term in range(k):
        idx = term + 1
        s_ax = fig.add_subplot(gs[1, pos])
        op = r"+ " if term else r"\,\,"
        s_ax.text(0.5, 0.5, rf"${op}s_{idx}$", ha="center")
        s_ax.set_axis_off()
        pos += 1
        u_term_ax = fig.add_subplot(gs[1, pos])
        u_term_ax.imshow(U[:, [term]], cmap=cmap)
        u_term_ax.text(-1.5, 1.0, rf"$\U_{idx}$", ha="right", va="bottom")
        pos += 2
        v_term_ax = fig.add_subplot(gs[0, pos])
        v_term_ax.imshow(Vt[[term], :], cmap=cmap)
        v_term_ax.text(0.5, -1.5, rf"$\V_{idx}$", ha="center", va="bottom")
        term_ax = fig.add_subplot(gs[1, pos])
        outer_prod = U[:, [term]].dot(Vt[[term], :])
        term_ax.imshow(outer_prod, cmap=cmap)
        pos += 1
    remainder_ax = fig.add_subplot(gs[1, pos])
    remainder_ax.text(0.5, 0.5, r"$+ \dots$", ha="center")
    remainder_ax.set_axis_off()


for i in range(1, 4):
    fig = show_decomposition(img, U, S, Vt, i)
    fig.savefig(FIGURES_DIR / f"pca_steps_{i}.png", transparent=True)
    fig.savefig(
        FIGURES_DIR / f"pca_steps_{i}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
