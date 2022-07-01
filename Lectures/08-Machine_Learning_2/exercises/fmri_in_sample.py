from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def load_abide_data(n_subjects=100):
    data = datasets.fetch_abide_pcp(
        n_subjects=n_subjects, derivatives=["rois_ho"]
    )
    X = data["rois_ho"]
    y = LabelEncoder().fit_transform(data["phenotypic"]["DX_GROUP"])
    return X, y


model = make_pipeline(
    ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    ),
    LogisticRegression(C=10),
)

X, y = load_abide_data()

model.fit(X, y)
predictions = model.predict(X)
score = accuracy_score(y, predictions)
print(f"\n\nPrediction accuracy: {score}")

scores = pd.DataFrame(cross_validate(model, X, y, return_train_score=True))
scores = scores.loc[:, ["train_score", "test_score"]].stack().reset_index()
scores.columns = ["split", "data", "score"]

sns.stripplot(data=scores, x="score", y="data")
plt.gca().set_xlabel("")
plt.gca().set_ylabel("")
plt.gca().set_title("Classification accuracy")
plt.tight_layout()
plt.show()
