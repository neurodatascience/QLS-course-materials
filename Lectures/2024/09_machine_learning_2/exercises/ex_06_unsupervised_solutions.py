# # Performing Clustering
# In this exercise, you will perform clustering on the handwritten digits data using k-means and hierarchical clustering. You will visualize the data with k-means and hierarchical clustering and answer some questions.
# ## Instructions
# - Load the the handwritten digits data
# - Visualize the data with PCA components.
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
    print("-"*50)
    cluster_order = np.unique(Z)
    print(cluster_order)
    print("-"*50)
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
    # plt.scatter(
    #     centroids[:, 0],
    #     centroids[:, 1],
    #     marker= "x",
    #     s=169,
    #     linewidths=3,
    #     color="w",
    #     zorder=10,
    # )
    for marker, center in zip(cluster_order, centroids):
        plt.text(center[0], center[1], str(marker), fontsize=30, color="white")

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


## Main exercise

X, y = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = X.shape, np.unique(y).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# apply PCA to reduce the data to 2D or 3D??
transformer = PCA(n_components=2)
X_pca = transformer.fit_transform(X)

# TODO - get rid of the loadings exercise
# PCA loadings
loadings = transformer.components_
# find the feature with the highest loading
print(
    f"The index of the feature with the highest loading in the first component is: {np.argmax(np.abs(loadings[0]))} with loading: {loadings[0][np.argmax(np.abs(loadings[0]))]}"
)
print(
    f"The index of the feature with the highest loading in the second component is: {np.argmax(np.abs(loadings[1]))} with loading: {loadings[1][np.argmax(np.abs(loadings[1]))]}"
)

# plot pca components with y labels in 2D
plt.figure()
for i in range(n_digits):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=i)
plt.legend()
plt.title("PCA components with y labels 2D")
plt.show()

# # plot 3D PCA components with y labels
# plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# ax = plt.axes(projection='3d')
# for i in range(n_digits):
#     ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=i)
# plt.legend()
# plt.title("PCA components with y labels 3D")
# plt.show()

# visualize the data with kmeans clustering
# play with the number of clusters
labels_pred_kmeans = visualize_kmeans(data=X_pca, n_clusters=n_digits)
plt.show()

# visualize the data with hierarchical clustering
# play with the number of clusters
# labels_pred_hierclstr = visualize_hclstr(data=X_pca, n_clusters=n_digits)
# plt.show()

# Clustering Performance Evaluation
# Here we will evaluate the performance of the clustering algorithms using ARI and the silhouette score.

print(f"Kmeans ARI score: {adjusted_rand_score(y, labels_pred_kmeans)}")
print(f"Kmeans Silhouette Score: {silhouette_score(X_pca, labels_pred_kmeans)}")

print(f"Hierarchical Clustering ARI score: {adjusted_rand_score(y, labels_pred_hierclstr)}")
print(f"Hierarchical Clustering Silhouette Score: {silhouette_score(X_pca, labels_pred_hierclstr)}")


# ## Additional exercise (optional)
# - Plot the ARI and silhouette scores for different number of clusters
# - Determine the optimal number of clusters for the data.
# - Compare the results with the number of clusters you have chosen.


# ## Questions
#
# - What do you observe when you visualize the data with k-means clustering?
#   - Answer: The data is clustered into 10 clusters in the sample space. The centroids are marked with white cross.
# - What do you observe when you visualize the data with hierarchical clustering?
#   - Answer: The dendrogram shows the hierarchy of clusters. The 10 clusters are specified with different colors.

# TODO
# - Do you think the data is well-suited for clustering? # How well does our data separate into clusters?
#   - Answer: No, the data is not well-suited for clustering as the data is not clearly separated into clusters.

# - What happened when you used a different number of clusters?
#   - Answer: The data was clustered into the number of clusters specified. For example, if we used 6 clusters, the data was still clustered into 6 clusters.

# TODO - clarify this is a sanity check and we usually don't have the true labels in "unsupervised" learning
# - What do you think is the best number of clusters for this data?
#   - Answer: The best number of clusters for this data is 10 based on our previous knowledge of the data.
# - (OPTIONAL) But is the number of clusters estimated from the data the same as the true number of clusters?
#   - Answer: ???
#
