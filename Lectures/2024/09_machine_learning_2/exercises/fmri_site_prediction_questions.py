# # Predicting and clustering site from fMRI: cross-validation and dimensionality reduction.
#
# Here we consider the same data as in the tutorial of
# Machine Learning part 1 of the QLS course.
#
# We had some fMRI time series, that we used to compute a connectivity matrix
# for each participant. We use the connectivity matrix values as our input
# features to predict to which site the participant belongs, as well as clustering
# the participants.
#
# ### Supervised Learning
#
# As in Part 1, we classify participants using a logistic regression. However
# we make several additions.
#
# ## Pipeline
#
# We use scikit-learn's `sklearn.pipeline.Pipeline`, that enables chaining
# several transformations into a single scikit-learn estimator (an object with
# a `fit` method). This ensures everything is fitted on the training data only --
# which is crucial here because we will add scaling a dimensionality reduction
# step with Principal Component Analysis.
#
# ## Scaling
#
# We add scaling of the input features using scikit-learn's StandardScaler,
# which removes the mean and scales the features to unit variance. This helps
# the logistic regression solver converge faster and often improves
# performance.
#
# ## Dimensionality Reduction
#
# We also consider a pipeline that reduces the dimension of input features with
# PCA, and compare it to the baseline logistic regrssion. One advantage is that
# the pipeline that uses PCA can be fitted much faster.
#
# ## Cross-validation
#
# In part 1 we fitted one model and evaluated it on a held-out test set. Here,
# we will use scikit-learn's `cross_validate` to perform K-Fold
# cross-validation and get a better estimate of our model's generalization
# performance. This allows comparing logistic regression with and without PCA,
# as well as a naive baseline.
#
# Moreover, instead of the plain `LogisticRegression`, we use scikit-learn's
# `LogisticRegressionCV`, which automatically performs a nested
# cross-validation loop on the training data to select the best hyperparameter.
#
# We therefore obtain a typical supervised learning experiment, with learning
# pipelines that involve chained transformations, hyperparameter selection, a
# cross-validation, and comparison of several models and a baseline.
#
# ### Unsupervised Learning
#
# In this script we will also use unsupervised learning to cluster the participants
# based on the connectivity features. We will use K-means and hierarchical clustering
# algorithms to cluster the participants and evaluate the performance of the clustering
# algorithms using ARI and the silhouette score.
#
#
# ## Exercises
#
# Read, understand and run this script. `load_data` loads the data
# and returns the matrices `X` and `y`. `prepare_pipelines` returns a
# dictionary whose values are scikit-learn estimators and whose keys are names
# for each estimator. All estimators are instances of scikit-learn's
# `Pipeline`.
#
# At the moment `prepare_pipelines` only returns 2 estimators: the logistic
# regression and a dummy estimator. Add a third estimator in the returned
# dictionary, which contains a dimensionality reduction step: a PCA with 20
# components. To do so, add a `sklearn.decomposition.PCA` as the second step of
# the pipeline. Note 20 is an arbitrary choice; how could we set the number of
# components in a principled way? What is the largest number of components we
# could ask for?
# Answer: include it in grid search, 80 (rank of X_train)
#
# There are 111 regions in the atlas we use to compute region-region
# connectivity matrices: the output of the `ConnectivityMeasure` has
# 111 * (111 - 1) / 2 = 6105 columns. If the dataset has 100 participants, What
# is the size of the coefficients of the logistic regression? of the selected
# (20 first) principal components? of the output of the PCA transformation (ie
# the compressed design matrix)?
# Answer: 6105 coefficients + intercept; principal components: 20 x 6105;
# compressed X: 100 x 20.
#
# Here we are storing data and model coefficients in arrays of 64-bit
# floating-point values, meaning each number takes 64 bits = 8 bytes of memory.
# Approximately how much memory is used by the design matrix X? by the
# dimensionality-reduced data (ie the kept left singular vectors of X)? by the
# principal components (the kept right singular vectors of X)?
# Answer: X: 4,884,000 B, compressed X: 16,000 B, V: 976,800 B
# (+ 96 bytes for all for the array object)
#
# As you can see, in this script we do not specify explicitly the metric
# functions that are used to evaluate models, but rely on scikit-learn's
# defaults instead. What metric is used in order to select the best
# hyperparameter? What metric is used to compute scores in `cross_validate`?
# Are these defaults appropriate for our particular situation?
# Answer: sklearn.metrics.accuracy_score for both, yes
#
# We do not specify the cross-validation strategy either. Which
# cross-validation procedure is used in `cross_validate`, and by the
# `LogisticRegressionCV`? Are these choices appropriate?
#
# ## Additional exercises (optional)
#
# Try replacing the default metrics with other scoring functions from
# scikit-learn or functions that you write yourself. Does the relative
# performance of the models change?
#
# Specify the cross-validation strategy explicitly, possibly choosing a
# different one than the default.
#
# Add another estimator to the options returned by `prepare_pipelines`, that
# uses univariate feature selection instead of PCA.
#
# What other approach could we use to obtain connectivity features of a lower
# dimension?
# Answer: use an atlas with less regions


import sys
import warnings
from logging import warning

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, "../../")

from data.utils import data_loader

warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_pipelines():
    """Prepare scikit-learn pipelines for fmri classification with connectivity.

    Returns a dictionary where each value is a scikit-learn estimator (a
    `Pipeline`) and the corresponding key is a descriptive string for that
    estimator.

    As an exercise you need to add a pipeline that performs dimensionality
    reduction with PCA.

    """
    scaling = StandardScaler()

    # Simple logistic regression
    logreg = LogisticRegression(C=10)

    # Fancier logistic regression with hyperparameter selection using internal grid search
    # logreg = LogisticRegressionCV(solver="liblinear", cv=3, Cs=3)

    logistic_reg = make_pipeline(clone(scaling), clone(logreg))
    # make_pipeline is a convenient way to create a Pipeline by passing the
    # steps as arguments. clone creates a copy of the input estimator, to avoid
    # sharing the state of an estimator across pipelines.
    pca_logistic_reg = make_pipeline(
        clone(scaling),
        PCA(n_components=20),
        clone(logreg),
    )
    kbest_logistic_reg = make_pipeline(
        clone(scaling),
        SelectKBest(f_classif, k=300),
        clone(logreg),
    )
    dummy = make_pipeline(DummyClassifier())
    # TODO: add a pipeline with a PCA dimensionality reduction step to this
    # dictionary. You will need to import `sklearn.decomposition.PCA`.
    return {
        "Logistic no PCA": logistic_reg,
        "Logistic with PCA": pca_logistic_reg,
        "Logistic with feature selection": kbest_logistic_reg,
        "Dummy": dummy,
    }


def compute_cv_scores(models, X, y):
    """Compute cross-validation scores for all models

    `models` is a dictionary like the one returned by `prepare_pipelines`, ie
    of the form `{"model_name": estimator}`, where `estimator` is a
    scikit-learn estimator.

    `X` and `y` are the design matrix and the outputs to predict.

    Returns a `pd.DataFrame` with one row for each model and cross-validation
    fold. Columns include `test_score` and `fit_time`.

    """
    all_scores = []
    for model_name, model in models.items():
        print(f"Computing scores for model: '{model_name}'")
        model_scores = pd.DataFrame(cross_validate(model, X, y, return_train_score=True))
        model_scores["model"] = model_name
        all_scores.append(model_scores)
    all_scores = pd.concat(all_scores)
    return all_scores


def visualize_kmeans(data, n_clusters):
    # reduce the data to 2D for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the data (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # find cluster labels using the original data
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    labels_pred = kmeans.fit_predict(data)

    return labels_pred


def visualize_hclstr(data, n_clusters):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(data)
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # find the color threshold to have n_clusters
    color_threshold = linkage_matrix[-n_clusters + 1, 2]

    # Plot the corresponding dendrogram
    plt.figure()
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, color_threshold=color_threshold)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")

    # find cluster labels
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels_pred = model.fit_predict(data)

    return labels_pred


if __name__ == "__main__":

    data, participants = data_loader()

    X = data.to_numpy()[:, 1:]

    label_col = "DX_GROUP"  # "SITE_ID" # "DX_GROUP"
    y = LabelEncoder().fit_transform(participants[label_col])

    # unique labels
    n_unique_labels = len(np.unique(y))
    print(f"The number of unique labels is {n_unique_labels}")

    # print data dimensions
    print(f"Data dimensions: {X.shape}")

    # plot explained variance ratio
    # what is the number of components that explains 90% of the variance?
    pca = PCA().fit(X)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.show()

    X_pca = PCA(n_components=20).fit_transform(X)

    # use select k best instead of PCA
    # X_selectK = SelectKBest(f_classif, k=300).fit_transform(X, y)

    # compare the data after applying PCA and select K best features
    # plot pca components with y labels
    plt.figure()
    for i in range(n_unique_labels):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f"Label {i}")
    plt.title("PCA components")
    plt.legend()

    # plot K best features with y labels
    # plt.figure()
    # for i in range(3):
    #     plt.scatter(X_selectK[y == i, 0], X_selectK[y == i, 1], label=f"Site {i}")
    # plt.title("Select K best features")
    # plt.legend()
    # plt.show()

    # # do the comparison in 3D
    # # PCA
    # plt.figure()
    # from mpl_toolkits.mplot3d import Axes3D
    # ax = plt.axes(projection='3d')
    # for i in range(3):
    #     ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=f"Site {i}")
    # plt.title("PCA components")
    # plt.legend()
    # # select K best features
    # plt.figure()
    # from mpl_toolkits.mplot3d import Axes3D
    # ax = plt.axes(projection='3d')
    # for i in range(3):
    #     ax.scatter(X_selectK[y == i, 0], X_selectK[y == i, 1], X_selectK[y == i, 2], label=f"Site {i}")
    # plt.title("Select K best features")
    # plt.legend()
    # plt.show()

    ## Classification
    # Predicting site from fMRI: cross-validation and dimensionality reduction.
    models = prepare_pipelines()
    all_scores = compute_cv_scores(models, X, y)
    print(all_scores.groupby("model").mean())
    # sns.stripplot(data=all_scores, x="train_score", y="model")
    # plt.tight_layout()
    # plt.show()

    ## Clustering

    # compare the performance of the models
    # try playing with the number of clusters
    # plot the data after applying kmeans clustering
    n_clusters = 2
    labels_pred_kmeans = visualize_kmeans(data=X_pca, n_clusters=n_clusters)
    # plot the data after applying hierarchical clustering
    labels_pred_hierclstr = visualize_hclstr(data=X_pca, n_clusters=n_clusters)

    plt.show()

    # Clustering Performance Evaluation
    # Here we will evaluate the performance of the clustering algorithms using ARI and the silhouette score.

    print(f"Kmeans ARI score: {adjusted_rand_score(y, labels_pred_kmeans)}")
    print(f"Kmeans Silhouette Score: {silhouette_score(X_pca, labels_pred_kmeans)}")

    print(f"Hierarchical Clustering ARI score: {adjusted_rand_score(y, labels_pred_hierclstr)}")
    print(
        f"Hierarchical Clustering Silhouette Score: {silhouette_score(X_pca, labels_pred_hierclstr)}"
    )
