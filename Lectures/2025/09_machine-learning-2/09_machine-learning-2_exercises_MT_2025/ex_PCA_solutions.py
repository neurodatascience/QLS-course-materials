# # Performing Principal Component Analysis (PCA) on the handwritten digits data
# ## Instructions
# - Load the the handwritten digits data
# - Visualize the data samples
# - Visualize the data with PCA components.
# - Create a pipeline with PCA and Logistic Regression to classify the data
# ## Hints
# - Use the `PCA` class from `sklearn.decomposition` to reduce the dimensionality of the data.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

## Main exercise

X, y = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = X.shape, np.unique(y).size

## Show the first 4 sample images
plt.figure(figsize=(8, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    x_img_sample = X[i].reshape(8, 8)
    plt.imshow(x_img_sample, cmap="gray")
    plt.title("y label: " + str(y[i]))
plt.show()

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# Apply PCA to reduce the data to 3D
transformer = PCA(n_components=3)
# Fit the transformer to the data and transform it
# and put the result in X_pca
X_pca = transformer.fit_transform(X)

# Plot pca components with y labels in 2D
plt.figure()
for i in range(n_digits):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=i)
plt.legend()
plt.title("PCA components with y labels 2D")
plt.show()

# Plot 3D PCA components with y labels
plt.figure()
from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection="3d")
for i in range(n_digits):
    ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], label=i)
plt.legend()
plt.title("PCA components with y labels 3D")
plt.show()


# Now let's use the PCA components to CLASSIFY the data

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a pipeline with PCA and Logistic Regression
# Choose the number of components in PCA such that 90% of variance is explained
# For Logistic Regression, use max_iter=1000 to avoid convergence warnings
pipeline = make_pipeline(PCA(n_components=0.9), LogisticRegression(max_iter=1000))

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# What is the number of components chosen?
# Find the attribute of the PCA that gives the number of components
# and replace ? with it
n_components = pipeline.named_steps["pca"].n_components_
print(f"Number of components: {n_components}")

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Find the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
