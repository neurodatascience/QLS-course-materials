# # Predicting and clustering site from fMRI
#
# We are given connectivity values derived from fMRI data
# for each participant. We use the connectivity values as our input
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
# which is crucial here because we will add scaling and dimensionality reduction
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
# Our pipeline also reduces the dimension of input features with PCA.
#
# ## Cross-validation
#
# In part 1 we fitted one model and evaluated it on a held-out test set. Here,
# we will use scikit-learn's `cross_validate` to perform K-Fold
# cross-validation and get a better estimate of our model's generalization
# performance.
#
# We therefore obtain a typical supervised learning experiment, with learning
# pipelines that involve chained transformations, a
# cross-validation, and comparison of models.
#
# ### Unsupervised Learning
#
# In this script we will also use unsupervised learning to cluster the participants
# based on the connectivity features. We will use K-means
# algorithms to cluster the participants and evaluate the performance of the
# algorithm using ARI and the silhouette score.
#
#
# ## Exercises
#
# Read, understand and run this script. `load_data` loads the data
# and returns the matrices `X` and `y`.
#
# As you can see, in this script we do not specify explicitly the metric
# functions that are used to evaluate models, but rely on scikit-learn's
# defaults instead. What metric is used to compute scores in `cross_validate`?
# Are these defaults appropriate for our particular situation?
#
# We do not specify the cross-validation strategy either. Which
# cross-validation procedure is used in `cross_validate`? Is this choice appropriate?


import sys
import warnings
from logging import warning

import numpy as np
from data.utils import data_loader
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# sys.path.insert(0, "../../")


warnings.filterwarnings("ignore", category=FutureWarning)


def load_data():
    """
    This function loads the fMRI site data and connectivity features
    and returns the features and labels.
    X: array-like, shape (n_samples, n_features)
    y: array-like, shape (n_samples,)
    """
    data, participants = data_loader()

    X = data.to_numpy()[:, 1:]

    label_col = "SITE_ID"  # "SITE_ID" # "DX_GROUP"
    y = LabelEncoder().fit_transform(participants[label_col])

    return X, y


def plot_clustering_evaluation_scores(
    X,
    y,
    n_clusters_range,
    metric,
):
    """
    This function plots the ARI and silhouette scores for different number of clusters
    n_clusters_range: range, the range of number of clusters to evaluate
    metric: str, the metric to use, either "ARI" or "silhouette"
    """

    kmeans_scores = []
    for n_clusters in n_clusters_range:
        labels_pred = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5).fit_predict(X)
        if metric == "ARI":
            kmeans_scores.append(adjusted_rand_score(y, labels_pred))
        elif metric == "silhouette":
            kmeans_scores.append(silhouette_score(X, labels_pred))

    plt.figure()
    plt.plot(n_clusters_range, kmeans_scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.title(f"Performance of clustering using {metric} score")
    plt.show()


if __name__ == "__main__":

    ## Load the data
    X, y = load_data()

    # unique labels
    n_unique_labels = len(np.unique(y))
    print(f"The number of unique labels is {n_unique_labels}")

    # print data dimensions
    print(f"Data dimensions: number of samples: {X.shape[0]}, number of features: {X.shape[1]}")

    ## Visualize the data

    # Apply PCA to the data to make it visualizable!
    X_pca = PCA(n_components=3).fit_transform(X)

    # Visualize the data after applying PCA
    # plot the first 2 PCs with y labels
    plt.figure()
    for i in range(n_unique_labels):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f"Site {i}")
    plt.title("PCA components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

    # Visualize the PCs in 3D
    # plot the first 3 PCs with y labels
    plt.figure(figsize=(10, 10))
    from mpl_toolkits.mplot3d import Axes3D

    ax = plt.axes(projection="3d")
    for i in range(n_unique_labels):
        ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=f"Site {i}", s=50)
    plt.title("PCA components")
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

    ## Classification

    # Predicting site from connectivity features:
    #   normalization
    #   dimensionality reduction
    #   classification

    # make_pipeline is a convenient way to create a Pipeline by passing the
    # steps as arguments.
    # StandardScaler is used to normalize the data before applying other steps
    # We create two pipelines:
    #   1. without PCA
    #   2. with PCA
    # and compare the performance of the two models

    # The first pipeline is a simple logistic regression model
    # model_without_pca = make_pipeline(
    #     StandardScaler(),
    #     LogisticRegression(C=10, max_iter=1000, solver="liblinear"),
    # )

    # The second pipeline is a logistic regression model with PCA
    # Set the number of components such that 90% of the variance is explained
    # model_with_pca = make_pipeline(?)

    # Cross validate the models
    # Use `cross_validate` from sklearn to evaluate the models
    # Set the `return_train_score` to True to get the train score
    # results_without_pca = ?
    # results_with_pca = ?

    # Print the results
    # print(
    #     "Cross-validation result for the model without PCA: "
    #     f"average train score: {results_without_pca['train_score'].mean():.2f} and "
    #     f"average test score: {results_without_pca['test_score'].mean():.2f}."
    # )
    # print(
    #     "Cross-validation result for the model with PCA: "
    #     f"average train score: {results_with_pca['train_score'].mean():.2f} and "
    #     f"average test score: {results_with_pca['test_score'].mean():.2f}."
    # )

    ## Clustering

    # First, apply PCA to reduce the data dimensionality
    # you can play with the number of components
    # but for this exercise, choose
    # the number of components such that 90% of the variance is explained
    # transformer = PCA(n_components=?)
    # X_pca = ?

    # find cluster labels
    # kmeans = ?
    # labels_pred = ?

    # Clustering Performance Evaluation
    # Here we will evaluate the performance of the clustering algorithm using ARI and the silhouette score.
    # print(f"Kmeans ARI score: {?}")
    # print(f"Kmeans Silhouette Score: {?}")

    # find the best number of clusters using ARI and silhouette score by
    # plotting the scores for different number of clusters using plot_clustering_evaluation_scores function
    # Is the best number of clusters clear from the plots?
    #
    #


# QUESTIONS
# What is the optimal number of clusters for the data based on the ARI score? What is the optimal number of clusters based on the silhouette score? Are the answers clear? Why?
# Why are the ARI and silhouette scores so poor even for the true number of clusters?
