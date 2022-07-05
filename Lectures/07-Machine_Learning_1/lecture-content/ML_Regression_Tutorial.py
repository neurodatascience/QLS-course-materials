# # Getting started with supervised learning: linear regression
#
# We use a small dataset distributed with scikit-learn, containing information
# about diabetic patients. 10 features were collected: age, sex, body mass
# index, average blood pressure, and six blood serum measurements. The target
# variable to predict is a continuous value measuring the progression of
# diabetes one year after the features were collected.
#
# To predict the disease progress, we will use a linear regression --
# implemented by `sklearn.linear_model.LinearRegression`. We will compare it to
# "random", "dummy", and "oracle" models.

# +
import numpy as np

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

# -

# We now load the dataset in memory as 2 numpy arrays `X` and `y`. `X` has
# shape `n_samples, n_features = 442, 10`, and `y` has shape `n_samples = 442`.
#
# We split the data into training and test examples, keeping 80% for training
# and 20% for testing.

# +

print('\n----------------------------------------------------------------------')
print('- 1) Load dataset ----------------------------------------------------')

X, y = datasets.load_diabetes(return_X_y=True)
(n_samples,n_features) = X.shape

print('\nUsing diabetes dataset from scikit-learn:'
'\nhttps://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes')

print(f'\nn_samples: {n_samples}, n_features: {n_features}')

n_train = int(len(y) * 0.8)
n_test = n_samples - n_train

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print(f'n_train samples: {n_train}, n_test samples: {n_test}')
# -

# Let's use a linear regression model for our prediction task. Note that the numbers here denote the features. 
# y_hat = b0 + b1*x1 + b2*x2 + b3*x3 ... + b10*x10

# **Exercise**: what will be the size of the model's coefficients (excluding
# the intercept)?

# +

print('\n----------------------------------------------------------------------')
print('- 2) Try a random model ----------------------------------------------')

# Let start with random initial model intercept (b0) and parameters (aka weights aka coefficients): b1,b2,b3 
initial_model_coefs = np.random.uniform(0,1, n_features)
initial_model_intercept = np.random.uniform(0,1,1)

# How do we calculate y_hat?
# For first sample: y_hat[0] = initial_model_intercept + initial_model_params[0]*X[0,0] + initial_model_params[1]*X[0,1] ... 
# For second sample: y_hat[1] = initial_model_intercept + initial_model_params[1]*X[1,0] + initial_model_params[1]*X[1,1] ... 

# TODO
# **Exercise**: write code to calculate y_hat for all training samples! 
# y_train_hat = 

# What is the error between model prediction (y_hat) and true values?
# initial_mse = np.mean((y_train_hat - y_train)**2)

# ------ Uncomment lines in this block to print results ------- 

# print(f'\nThese are the (random) model parameters before training:')
# print(f'b0 (intercept): {initial_model_intercept}\nb1-b10: {initial_model_coefs}')
# print(f'\nMSE before training (i.e. using random weights): {initial_mse:.5g}')

# -------------------------------------------------------------

#-

print('\n----------------------------------------------------------------------')
print('- 3) Use scikit-learn to fit the model -------------------------------')

# Now let's use scikit-learn to fit the model to the training data.
# +

model = LinearRegression()
# TODO
# **Exercise**: fit the model with training data and get the predictions. 

# What did our model learn? 
# Let's look at the model parameters (aka weights aka coefficients): b1,b2,b3 ... 
# Note that sklearn saves the b0 separately as intercept. 

# trained_model_intercept = model.intercept_
# trained_model_coef = model.coef_


# TODO
# **Exercise**: Check if our MSE calculation matches with sklearn! 
# y_train_hat = 
# my_train_mse = 

# ------ Uncomment lines in this block to print results ------- 

# print(f'\nThese are the model parameters after training:')
# print(f'b0 (intercept): {trained_model_intercept}\nb1-b10: {trained_model_coef}')
# print(f'\nMy calculation of MSE after model training (on train data): {my_train_mse:.5g}')
# print(f"\nScikit-learn's calculation of MSE after model training (on train data): {train_mse:.5g}")

# -------------------------------------------------------------

# -


print('\n----------------------------------------------------------------------')
print('- 4) Evaluate our model on (unseen) test data ------------------------')

# Now we evaluate our model on (unseen) test data and display its Mean Squared Error.
# +

# TODO
# **Exercise**: Check test set performance 
# test_predictions = 
# test_mse = mean_squared_error(y_test, test_predictions)

# ------ Uncomment lines in this block to print results ------- 

# print('\nBut what matter is the model performance on the test set!!')
# print(f"Mean squared error on test data: {test_mse:.5g}")

# -------------------------------------------------------------

print('\n----------------------------------------------------------------------')
print('- 5) Compare our performance against dummy and oracle models ---------')

# We train a `DummyRegressor`. This estimator makes a constant prediction (it
# ignores the features and always predicts the same value for `y`). However,
# this constant value is not arbitrary: it is the one that results in the
# smallest Mean Squared Error on the training data.
#
# **Exercise**: what constant value prediction minimizes the MSE for the training sample?

# ------ Uncomment lines in this block to print results ------- 

# dummy_predictions = DummyRegressor().fit(X_train, y_train).predict(X_test)
# dummy_mse = mean_squared_error(y_test, dummy_predictions)
# print(f"\nMean squared error of dummy model on test data: {dummy_mse:.5g}")

# # What would be ideal predictions? (Impossible in real life!)
# # If we had a oracle predictor - it would predict the true values on the test set perfectly! 
# oracle_predictions = y_test
# oracle_mse = mean_squared_error(y_test, oracle_predictions)
# print(f"Mean squared error of oracle model on test data: {oracle_mse:.5g}")

# -------------------------------------------------------------

# -

# Finally, we display the true outcomes and the predictions of our models for
# the test data. Would you say the linear regression is doing much better than
# the dummy model?

# **Exercise**: Is it possible to do worse than the dummy model?

print('\n----------------------------------------------------------------------')
print('- 6) Make Plots! -----------------------------------------------------')

# ------ Uncomment lines in this block to print results ------- 
# plt.plot(
#     [y_test.min(), y_test.max()],
#     [y_test.min(), y_test.max()],
#     color="black",
#     linestyle="--",
# )
# plt.scatter(y_test, oracle_predictions,marker="d")
# plt.scatter(y_test, test_predictions)
# plt.scatter(y_test, dummy_predictions, marker="^")
# plt.legend(
#     [
#         "Identity line",
#         "Oracle model --> Perfect prediction",
#         f"LinearRegression model (MSE = {test_mse:.5g})",
#         f"DummyRegressor model (MSE = {dummy_mse:.5g})",
#     ]
# )
# plt.gca().set_xlabel("True outcome")
# plt.gca().set_ylabel("Predicted outcome")
# plt.gca().set_title("True and predicted diabetes progress")
# plt.show()

# -------------------------------------------------------------


# Here we have a small number of features that are not too correlated
# (condition number of `X_train` is 23), so linear regression without
# regularization works well. If the number of features were large, or if the
# columns of X were not linearly independent, what could we use to stabilize
# the model's parameters?

print('\n- Done!!!-------------------------------------------------------------\n')