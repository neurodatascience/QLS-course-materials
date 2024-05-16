from sklearn import model_selection
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

# import pickle

X, y = make_regression(noise=10, random_state=0)
model = Ridge()

# TODO: using an appropriate function from scikit-learn, compute
# cross-validation scores for a ridge regression on this dataset. What
# cross-validation strategy is used? what do the scores represent -- what
# performance metric is used?
# What is a good choice for k?
# Hint:
#  - see cross_validate: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
#  - or cross_val_score (very similar, a simpler interface to cross_validate): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
scores = "???"
# TODO_BEGIN
scores = model_selection.cross_validate(
    model, X, y, scoring="neg_mean_squared_error", cv=model_selection.KFold(5)
)
# TODO_END
fit_time = scores["fit_time"]
score_time = scores["score_time"]
test_score = scores["test_score"].round(2)

print(f"Fit time: {fit_time}")
print(f"Score time: {score_time}")
print(f"Test score (i.e. neg_mean_squared_error): {test_score}")

# if I am satisfied with scores
model.fit(X, y)
# with open("/tmp/model_ready_for_production.pkl", "wb") as f:
#     pickle.dump(model, f)
