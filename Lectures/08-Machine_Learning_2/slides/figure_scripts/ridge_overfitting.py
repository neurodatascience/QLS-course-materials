import time

import numpy as np
from sklearn import (
    datasets,
    linear_model,
    model_selection,
    base,
    metrics,
    feature_selection,
    pipeline,
)
from matplotlib import pyplot as plt

from config import FIGURES_DIR, TAB10_COLORS


def score(n_features, model, X_train, X_test, y_train, y_test):
    model = base.clone(model)
    X_train = X_train[:, :n_features]
    X_test = X_test[:, :n_features]
    start = time.perf_counter()
    model.fit(X_train, y_train)
    duration = time.perf_counter() - start
    train_score = metrics.mean_squared_error(y_train, model.predict(X_train))
    test_score = metrics.mean_squared_error(y_test, model.predict(X_test))
    return train_score, test_score, duration


def run(random_state, dimensions, dim_reduction=None):
    X, y = datasets.make_regression(
        n_samples=200,
        n_features=300,
        n_informative=3,
        shuffle=False,
        random_state=random_state,
        noise=1.0,
    )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=False, test_size=0.5
    )
    assert X_train.shape == (100, 300)
    ridge = linear_model.Ridge(0.000001, solver="cholesky")
    train_scores, test_scores = [], []
    durations = []
    for n_features in dimensions:
        if dim_reduction is not None:
            model = pipeline.Pipeline(
                [
                    (
                        "feature_selection",
                        feature_selection.SelectKBest(
                            feature_selection.f_regression,
                            k=min(n_features, 10),
                        ),
                    ),
                    ("ridge", ridge),
                ]
            )
        else:
            model = ridge
        train_s, test_s, t = score(
            n_features, model, X_train, X_test, y_train, y_test
        )
        train_scores.append(train_s)
        test_scores.append(test_s)
        durations.append(t)
    return train_scores, test_scores, durations


dimensions = list(range(5, 99))
n_runs = 30
all_train_scores, all_test_scores, all_durations = zip(
    *(run(i, dimensions) for i in range(n_runs))
)
train_scores = np.median(all_train_scores, axis=0)
test_scores = np.median(all_test_scores, axis=0)
durations = np.median(all_durations, axis=0)

all_train_scores_select, all_test_scores_select, all_durations_select = zip(
    *(run(i, dimensions, "kbest") for i in range(n_runs))
)
train_scores_select = np.median(all_train_scores_select, axis=0)
test_scores_select = np.median(all_test_scores_select, axis=0)
durations_select = np.median(all_durations_select, axis=0)

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(dimensions, train_scores, linestyle="--", color=TAB10_COLORS[0])
ax.plot(dimensions, test_scores, color=TAB10_COLORS[0])
ax.legend(
    ["Training Mean Squared Error", "Testing Mean Squared Error"],
    frameon=False,
)
# ax.set_yscale("log")
ax.set_xlabel("Number of features $p$\n(only 3 are actually informative)")
ax.set_title(
    f"Fitting a linear regression on 100 points"
    f"\n(median across {n_runs} simulations)"
)

fig.savefig(FIGURES_DIR / "mse.pdf", bbox_inches="tight")
ax.set_yscale("log")
fig.savefig(FIGURES_DIR / "mse_log.pdf", bbox_inches="tight")
ax.plot(dimensions, train_scores_select, linestyle="--", color=TAB10_COLORS[1])
ax.plot(dimensions, test_scores_select, color=TAB10_COLORS[1])
ax.legend(
    [
        "Training MSE no screening",
        "Testing MSE no screening",
        "Training MSE with screening",
        "Testing MSE with screening",
    ],
    frameon=False,
)
ax.set_yscale("linear")
fig.savefig(FIGURES_DIR / "mse_with_dim_reduction.pdf", bbox_inches="tight")
ax.set_yscale("log")
fig.savefig(
    FIGURES_DIR / "mse_with_dim_reduction_log.pdf", bbox_inches="tight"
)

plt.close("all")
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(dimensions, durations)
ax.set_xlabel("Number of features $p$")
ax.set_title(
    f"Fitting a linear regression on 100 points"
    f"\n(median across {n_runs} simulations)"
)
ax.set_ylabel("Time fitting model (s)")
fig.savefig(FIGURES_DIR / "durations.pdf", bbox_inches="tight")
