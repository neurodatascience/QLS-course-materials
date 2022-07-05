from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt

all_differences = []
for seed in range(100):
    X, y = make_regression(noise=10, n_features=1000, random_state=seed)

    X_reduced = SelectKBest(f_regression).fit_transform(X, y)
    overfit = cross_validate(Ridge(), X_reduced, y)["test_score"].mean()

    model = make_pipeline(SelectKBest(f_regression), Ridge())
    pipe = cross_validate(model, X, y)["test_score"].mean()
    all_differences.append(overfit - pipe)


plt.boxplot(
    all_differences,
    vert=False,
)
plt.gca().set_xlabel(
    "overfit score - correct score (averaged over 100 simulations)"
)
plt.tight_layout()
plt.show()
