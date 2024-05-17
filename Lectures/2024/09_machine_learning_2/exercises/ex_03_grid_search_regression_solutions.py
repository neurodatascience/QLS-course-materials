from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, cross_validate

X, y = make_regression(noise=2.5, n_samples=500, n_features=600, random_state=0)

# TODO: try a few different values of alpha
model = Ridge(alpha=1e4)

# TODO: instead of using a model with only one parameter, use `GridSearchCV` to
# create a meta-model that will select the best hyperparameter using a nested
# cross-validation loop.
# TODO_BEGIN

# hyperparameter search grid
alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

model = GridSearchCV(Ridge(), {"alpha": alpha_grid}, verbose=1)
# TODO_END
scores = cross_validate(model, X, y)

fit_time = scores["fit_time"]
score_time = scores["score_time"]
test_score = scores["test_score"].round(2)

print("-" * 50)
print("Default GridSearchCV with Ridge model:")
print("-" * 50)
print(f"Fit time: {fit_time}")
print(f"Score time: {score_time}")
print(f"Test score (i.e. neg_mean_squared_error): {test_score}")
print("-" * 50)
# scikit-learn also has a `RidgeCV` model that does this gridsearch much more
# efficiently.

ridgecv_model = RidgeCV(alphas=alpha_grid)
ridgecv_scores = cross_validate(ridgecv_model, X, y)

fit_time = ridgecv_scores["fit_time"]
score_time = ridgecv_scores["score_time"]
test_score = ridgecv_scores["test_score"].round(2)

print("More efficient RidgeCV model:")
print("-" * 50)
print(f"Fit time: {fit_time}")
print(f"Score time: {score_time}")
print(f"Test score (i.e. neg_mean_squared_error): {test_score}")
print("-" * 50)
