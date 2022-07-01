from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_validate, GridSearchCV

X, y = make_regression(noise=2.5, n_samples=500, n_features=600, random_state=0)

# TODO: try a few different values of alpha
model = Ridge(alpha=1e4)

# TODO: instead of using a model with only one parameter, use `GridSearchCV` to
# create a meta-model that will select the best hyperparameter using a nested
# cross-validation loop.
# TODO_BEGIN
model = GridSearchCV(Ridge(), {"alpha": [.1, 1., 10.]}, verbose=1)
# TODO_END
scores = cross_validate(model, X, y)
print("scores:", scores["test_score"])

# scikit-learn also has a `RidgeCV` model that does this gridsearch much more
# efficiently.

ridgecv_model = RidgeCV(alphas=[.001, .01, .1, 1., 10., 100., 1000.])
ridgecv_scores = cross_validate(ridgecv_model, X, y)
print("scores using RidgeCV:", ridgecv_scores["test_score"])
