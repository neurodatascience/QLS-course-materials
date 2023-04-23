from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt

X, y = make_regression(noise=10, n_features=5000, random_state=0)

# Question: what is the issue with the code below?

X_reduced = SelectKBest(f_regression).fit_transform(X, y)
scores = cross_validate(Ridge(), X_reduced, y)["test_score"]
print("feature selection in 'preprocessing':", scores)

# Now fitting the whole pipeline on the training set only

# TODO:
# - use `make_pipeline` to create a pipeline chaining a `SelectKBest` and a
#   `Ridge`
# - use `cross_validate` to obtain cross-validation scores for the whole
#   pipeline treated as a single model
# See: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
model = "make_pipeline(???)"
scores_pipe = ()
print("feature selection on train set:", scores_pipe)

plt.boxplot(
    [scores_pipe, scores],
    vert=False,
    labels=[
        "feature selection on train set",
        "feature selection on whole data",
    ],
)
plt.gca().set_xlabel("R² score")
plt.tight_layout()
plt.show()
