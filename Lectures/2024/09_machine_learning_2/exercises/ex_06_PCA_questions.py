# # Performing Principal Component Analysis (PCA) on the handwritten digits data
# ## Instructions
# - Load the the handwritten digits data
# - Visualize the data samples
# - Visualize the data with PCA components.
# ## Hints
# - Use the `PCA` class from `sklearn.decomposition` to reduce the dimensionality of the data.

import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import load_digits

## Main exercise

X, y = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = X.shape, np.unique(y).size


## Show the first 4 sample images
plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    x_img_sample = X[i].reshape(8, 8)
    plt.imshow(x_img_sample, cmap="gray")
    plt.title("y label: " + str(y[i]))
plt.show()

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# apply PCA to reduce the data to 3D
# X_pca = ?
#

# # plot pca components with y labels in 2D
# plt.figure()
# for i in range(n_digits):
#     plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=i)
# plt.legend()
# plt.title("PCA components with y labels 2D")
# plt.show()

# # plot 3D PCA components with y labels
# plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# ax = plt.axes(projection='3d')
# for i in range(n_digits):
#     ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=i)
# plt.legend()
# plt.title("PCA components with y labels 3D")
# plt.show()
