import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.base import clone
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt


def dof(X, alpha_range):
    U, S, Vt = svd(X)
    return np.sum(S ** 2 / (S ** 2 + alpha_range[:, None]), axis=1)


alpha_range = np.logspace(-3, 3, 50)

# model = RidgeCV(alphas=alpha_range, store_cv_values=True)
X, y, true_coef = make_regression(
    noise=5,
    effective_rank=2,
    random_state=0,
    coef=True,
    shuffle=False,
    n_samples=100,
)
train = np.arange(len(y) // 2)
test = np.arange(len(y) // 2, len(y))
print(true_coef[:20])
# fitted = clone(model).fit(X, y)
# mse = fitted.cv_values_.mean(axis=0)
# r2 = 1 - fitted.cv_values_.sum(axis=0) / (len(y) * y.var())
coef = []
train_mse = []
test_mse = []
for alpha in alpha_range:
    ridge = Ridge(alpha, fit_intercept=True).fit(X[train], y[train])
    train_mse.append(mean_squared_error(y[train], ridge.predict(X[train])))
    test_mse.append(mean_squared_error(y[test], ridge.predict(X[test])))
    coef.append(ridge.coef_)
coef = np.asarray(coef)
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
# test_r2 = 1 - test_mse / y[test].var()
ax[0].plot(alpha_range, test_mse)
ax[0].plot(alpha_range, train_mse)
ax[0].legend(["testing error", "training error"])
ax[0].set_ylabel("Mean squared error")
ax[1].plot(alpha_range, test_mse)
ax[1].set_ylabel("Mean squared error")
ax[1].legend(["testing error"], loc="lower right")
ax[2].plot(alpha_range, coef[:, 0:50:5])
# ax.axhline(true_coef[1])
ax[2].set_xscale("log")
ax[2].set_ylabel("Example coefficients")
ax[2].set_xlabel("Regularization hyperparameter ɑ")
ax[0].set_title(
    "Hyperparameter choice matters: Ridge regression\n"
    "MSE and coefficients for a range of ɑ"
)
plt.tight_layout()
plt.gcf().savefig("hyperparameter_selection.pdf", bbox_inches="tight")
plt.show()
