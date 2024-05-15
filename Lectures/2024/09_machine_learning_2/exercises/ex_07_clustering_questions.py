# # Performing Clustering
# In this exercise, you will perform clustering on the handwritten digits data using k-means and hierarchical clustering. You will visualize the data with k-means and hierarchical clustering and answer some questions.
# ## Instructions
# - Load the the handwritten digits data
# - Reduce the dimensionality of the data with PCA.
# - Visualize the data with k-means clustering.
# - Visualize the data with hierarchical clustering.
# - Answer the questions.
# ## Hints
# - Use the `PCA` class from `sklearn.decomposition` to reduce the dimensionality of the data.
# - Use the `visualize_kmeans` function to perform and visualize k-means clustering.
# - Use the `visualize_hclstr` function to perform and visualize hierarchical clustering.
# - Use the `adjusted_rand_score` and `silhouette_score` functions from `sklearn.metrics` to evaluate the clustering performance.


import matplotlib.pyplot as plt

# +
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

# -


# ## Utilities
#
# The functions below are helpers for clustering and visualization.
# You should read them but they do not need to be modified.


def visualize_kmeans(data, n_clusters):
    '''
    This function performs k-means clustering and visualizes the data with a decision boundary.
    data: array-like, shape (n_samples, n_features)
    n_clusters: int, number of clusters
    '''
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
    # Probably want to sort this to get sorted cluster order

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
        marker= "x",
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
    '''
    This function performs hierarchical clustering and visualizes the data with a dendrogram.
    data: array-like, shape (n_samples, n_features)
    n_clusters: int, number of clusters
    '''
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

def plot_clustering_evaluation_scores(
    X, y, n_clusters_range, metric,
):
    '''
    This function plots the ARI and silhouette scores for different number of clusters
    n_clusters_range: range, the range of number of clusters to evaluate
    metric: str, the metric to use, either "ARI" or "silhouette"
    '''

    kmeans_scores = []
    hierclstr_scores = []
    for n_clusters in n_clusters_range:
        labels_pred_kmeans = KMeans(n_clusters=n_clusters).fit_predict(X)
        labels_pred_hierclstr = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
        if metric == "ARI":
            kmeans_scores.append(adjusted_rand_score(y, labels_pred_kmeans))
            hierclstr_scores.append(adjusted_rand_score(y, labels_pred_hierclstr))
        elif metric == "silhouette":
            kmeans_scores.append(silhouette_score(X, labels_pred_kmeans))
            hierclstr_scores.append(silhouette_score(X, labels_pred_hierclstr))

    plt.figure()
    plt.plot(n_clusters_range, kmeans_scores, label="kmeans")
    plt.plot(n_clusters_range, hierclstr_scores, label="hirarchical")
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Performance of clustering using {metric} score")
    plt.show()


## Main exercise

X, y = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = X.shape, np.unique(y).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# apply PCA to reduce the data dimensionality
# you can play with the number of components
# but for this exercise, we will use number of components = n_digits
transformer = PCA(n_components=n_digits)
X_pca = transformer.fit_transform(X)

# visualize the data with kmeans clustering
# play with the number of clusters
# labels_pred_kmeans = ?
# plt.show()

# visualize the data with hierarchical clustering
# play with the number of clusters
# labels_pred_hierclstr = ?
# plt.show()

# Clustering Performance Evaluation
# Here we will evaluate the performance of the clustering algorithms using ARI and the silhouette score.

# print(f"Kmeans ARI score: {?}")
# print(f"Kmeans Silhouette Score: {?}")

# print(f"Hierarchical Clustering ARI score: {?}")
# print(f"Hierarchical Clustering Silhouette Score: {?}")


# ## Additional exercise (optional)
# - Plot the ARI and silhouette scores for different number of clusters
# - Determine the optimal number of clusters for the data.
# - Compare the results with the number of clusters you have chosen.



# ## Questions
#
# - What do you observe when you visualize the data with k-means clustering?
# - What do you observe when you visualize the data with hierarchical clustering?
# - How well does our data separate into clusters?
# - What happened when you used a different number of clusters than n_digits?
# - What do you think is the best number of clusters for this data?
# - (OPTIONAL) Is the number of clusters estimated from the data the same as the true number of clusters?
