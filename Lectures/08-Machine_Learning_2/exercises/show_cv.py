import argparse

import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from matplotlib import gridspec, colors

EMPTY_COLOR = "#ffffff"
TRAIN_CMAP = colors.ListedColormap(
    [EMPTY_COLOR, colors.TABLEAU_COLORS["tab:blue"]]
)
TEST_CMAP = colors.ListedColormap(
    [EMPTY_COLOR, colors.TABLEAU_COLORS["tab:orange"]]
)


def show_nested_cv(outer_k, inner_k):
    n_splits = outer_k * (2 + inner_k * 2)
    fig = plt.figure(figsize=(12, n_splits / 3))
    outer_splits = KFold(outer_k).split(np.arange(100))
    grid = fig.add_gridspec(outer_k, 1, hspace=0.1, wspace=0)
    for i, (train, test) in enumerate(outer_splits):
        show_outer_fold(i, inner_k, grid, fig, outer_train_idx=train)
    return fig


def add_box(subplotspec, fig, text=""):
    ax = fig.add_subplot(subplotspec)
    if text != "":
        ax.text(
            0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes
        )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def show_outer_fold(fold_idx, inner_k, outer_grid, fig, outer_train_idx):
    fold_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        7,
        outer_grid[fold_idx, :],
        wspace=0,
        hspace=0,
        height_ratios=[2 * inner_k + 1, 1],
    )
    add_box(fold_grid[:, 0], fig, f"Fold {fold_idx}")
    add_box(fold_grid[0, 1], fig, "Train")
    add_box(fold_grid[1, 1:-2], fig, "Test")
    test_ax = add_box(fold_grid[-1, -2], fig)
    train_mask = np.zeros((1, 100), dtype=int)
    train_mask[:, outer_train_idx] = 1
    test_ax.imshow(~train_mask, aspect="auto", cmap=TEST_CMAP)
    score_ax = fig.add_subplot(fold_grid[-1, -1])
    score_ax.text(
        0.1,
        0.5,
        f"Score {fold_idx}",
        ha="left",
        va="center",
        transform=score_ax.transAxes,
    )
    score_ax.axis("off")
    show_inner_cv(
        fold_idx,
        fold_grid,
        fig,
        inner_k=inner_k,
        outer_train_idx=outer_train_idx,
    )


def show_inner_cv(fold_idx, fold_grid, fig, inner_k, outer_train_idx):
    nested_cv_grid = gridspec.GridSpecFromSubplotSpec(
        (inner_k * 2) + 1, 4, fold_grid[:-1, 2:-1], wspace=0, hspace=0
    )
    for i in range(inner_k):
        add_box(nested_cv_grid[2 * i : 2 * i + 2, 0], fig, f"Fold {i}")
        add_box(nested_cv_grid[2 * i, 1], fig, "Train")
        add_box(nested_cv_grid[2 * i + 1, 1], fig, "Test")
        add_box(nested_cv_grid[2 * i, 2], fig, "For all λ")
        add_box(nested_cv_grid[2 * i + 1, 2], fig, "For all λ")
    add_box(nested_cv_grid[-1, :2], fig, "Refit")
    add_box(nested_cv_grid[-1, 2], fig, "For best λ")
    show_inner_splits(inner_k, nested_cv_grid, fig, outer_train_idx)


def show_inner_splits(inner_k, nested_cv_grid, fig, outer_train_idx):
    mask = np.zeros((1, 100), dtype=int)
    inner_splits = KFold(inner_k).split(outer_train_idx)
    for i, (train, test) in enumerate(inner_splits):
        mask[:] = 0
        mask[:, outer_train_idx[train]] = 1
        train_ax = add_box(nested_cv_grid[2 * i, -1], fig)
        train_ax.imshow(mask, aspect="auto", cmap=TRAIN_CMAP)
        mask[:] = 0
        mask[:, outer_train_idx[test]] = 1
        test_ax = add_box(nested_cv_grid[2 * i + 1, -1], fig)
        test_ax.imshow(mask, aspect="auto", cmap=TEST_CMAP)
    refit_ax = add_box(nested_cv_grid[-1, -1], fig)
    mask[:] = 0
    mask[:, outer_train_idx] = 1
    refit_ax.imshow(mask, aspect="auto", cmap=TRAIN_CMAP)


def show_simple_cv(k):
    fig = plt.figure(figsize=(6, k))
    splits = KFold(k).split(np.arange(100))
    grid = fig.add_gridspec(2 * k, 4, hspace=0, wspace=0)
    for i, (train, test) in enumerate(splits):
        add_box(grid[2 * i : 2 * i + 2, 0], fig, f"Fold {i}")
        add_box(grid[2 * i, 1], fig, "Train")
        add_box(grid[2 * i + 1, 1], fig, "Test")
        train_mask = np.zeros((1, 100), dtype=int)
        train_mask[:, train] = 1
        train_ax = add_box(grid[2 * i, 2], fig)
        train_ax.imshow(train_mask, aspect="auto", cmap=TRAIN_CMAP)
        test_ax = add_box(grid[2 * i + 1, 2], fig)
        test_ax.imshow(~train_mask, aspect="auto", cmap=TEST_CMAP)
        score_ax = fig.add_subplot(grid[2 * i + 1, -1])
        score_ax.text(
            0.1,
            0.5,
            f"Score {i}",
            ha="left",
            va="center",
            transform=score_ax.transAxes,
        )
        score_ax.axis("off")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str, default=None)
    parser.add_argument(
        "--outer_k", type=int, default=5, help="Number of folds in outer loop"
    )
    parser.add_argument(
        "--inner_k",
        type=int,
        default=3,
        help="Number of folds in inner loop "
        "(used for hyperparameter selection)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Show simple K-fold (without grid search)",
    )
    args = parser.parse_args()
    if args.simple:
        fig = show_simple_cv(args.outer_k)
    else:
        fig = show_nested_cv(args.outer_k, args.inner_k)
    fig.savefig(args.output_file, bbox_inches="tight")
