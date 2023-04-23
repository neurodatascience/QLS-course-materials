from sklearn.datasets import make_regression
from sklearn import model_selection
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
print(f"\nCross-validation result:\n\n{scores}")

# if I am satisfied with scores
model.fit(X, y)
# with open("/tmp/model_ready_for_production.pkl", "wb") as f:
#     pickle.dump(model, f)
