# # Performing nested cross-validation
#
# Here we will evaluate the performance of an L2-regularized logistic
# regression on one of the classification datasets distributed with
# scikit-learn.
#
# The model has a hyperparameter C, which controls the regularization strength:
# a higher value of C means less regularization. We will automatically select
# the appropriate value for C among a grid of possible values with a nested
# cross-validation loop.
#
# The whole procedure therefore looks like:
#
# - Outer loop:
#   - obtain 5 (train, test) splits for the whole dataset
#   - initialize `all_scores` to an empty list
#   - for each (train, test) split:
#     + run grid-search (ie inner CV loop) on training data and obtain a model
#       fitted to the whole training data with the best hyperparameter.
#     + evaluate the model on the test data
#     + append the resulting score to `all_scores`
#   - return `all_scores`
#
# - Grid-search (inner loop):
#   - obtain 3 (train, test) splits for the available data (the training data
#     from the outer loop)
#   - initialize `all_scores` to an empty list
#   - for each possible hyperparameter value C:
#     + initialize `scores_for_this_C` to an empty list
#     + for each  train, test split:
#       * fit a model on train, using the hyperparameter C
#       * evaluate the model on test
#       * append the resulting score to `scores_for_this_C`
#     + append the mean of `scores_for_this_C` to `all_scores`
#   - select the hyperparameter with the best mean score
#   - refit the model on the whole available data, using the selected
#     hyperparameter
#   - return this model
#
# Most of this logic is implemented in this module, but some key parts are
# still missing (marked with "TODO"). Your job is to complete the functions
# `cross_validate` and `grid_search` so that the whole nested cross-validation
# can be run.
#
# Some helper routines, `get_kfold_splits` and `fit_and_score`, are
# provided to make the task easier. Make sure you read their code and
# understand what they do.
#
# The docstrings of the incomplete functions document precisely what are their
# parameters, and what they should compute and return. Rely on this information
# to write your implementation.
#
# This nested cross-validation procedure is often used, so scikit-learn
# provides all the functionality we are implementing here. To check that our
# implementation is correct, we can therefore compare results with what we
# obtain from scikit-learn. At the end of the file, you will see code that
# loads a scikit-learn dataset, and computes cross-validation scores using our
# `cross_validate` function, then using scikit-learn, and prints both results.
# If you execute this script by running `python nested_cross_validation.py`,
# these two results will be shown and you can check that the code runs and
# produces correct results.

# +
import numpy as np
from sklearn import datasets, linear_model, model_selection, metrics
from sklearn.base import clone

# -


# ## Utilities
#
# The functions below are helpers for the main routines `cross_validate` and
# `grid_search`. You should read them but they do not need to be modified.


def load_data():
    """Load iris data

    This function shuffles the data (without breaking the pairing of X and y)
    so that the examples are not sorted by class.

    Returns
    -------
    tuple of length 2
      X, design matrix of shape (n_samples, n_features)
      y, targets, of shape (n_samples,)
    """
    X, y = datasets.load_iris(return_X_y=True)
    idx = np.arange(len(y))
    np.random.RandomState(0).shuffle(idx)
    # this is now the recommended way of doing this, but only works with recent
    # versions of numpy:
    # np.random.default_rng(0).shuffle(idx)
    X, y = X[idx], y[idx]
    return X, y


def get_kfold_splits(n_samples, k):
    """Given a total number of samples, return k-fold (train, test) indices.

    Parameters
    ----------
    n_samples : int
      The total number of samples in the dataset to be split for
      cross-validation.

    k : int, optional
      The number of cross-validation folds

    Returns
    -------
    splits : list[tuple[np.array[int], np.array[int]]]
      each element of `splits` corresponds to one cross-validation fold and
      contains a pair of arrays (train, test):
      - train: the integer indices of samples in the training set
      - test: the integer indices of samples in the testing set

    """
    indices = np.arange(n_samples)
    test_mask = np.empty(n_samples, dtype=bool)
    splits = []
    start = 0
    for i in range(k):
        n_test = n_samples // k
        if i < n_samples % k:
            n_test += 1
        stop = start + n_test
        test_mask[:] = False
        test_mask[start:stop] = True
        splits.append((indices[np.logical_not(test_mask)], indices[test_mask]))
        start = stop
    return splits


def fit_and_score(model, C, X, y, train_idx, test_idx, score_fun):
    """Fit a model on training data and compute its score on test data.

    Parameters
    ----------
    model : scikit-learn estimator (will not be modified)
      the estimator to be evaluated

    C : float
     The value for the regularization hyperparameter C to use when fitting the
     model.

    X : numpy array of shape (n_samples, n_features)
      the full design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the full target vector

    train_idx : sequence of ints
      the indices of training samples (row indices of X)

    test_idx : sequence of ints
      the indices of testing samples

    score_fun : callable
      the function that measures performance on test data, with signature
     `score = score_fun(true_y, predicted_y)`.

    Returns
    -------
      The prediction score on test data

    """
    model = clone(model)
    model.set_params(C=C)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    score = score_fun(y[test_idx], predictions)
    print(f"    Inner CV loop: fit and evaluate one model; score = {score:.2f}")
    return score


# ## Exercises
#
# The two functions below are incomplete! Complete the body of each function so
# that it behaves as described in the docstring.


def grid_search(model, C_candidates, X, y, inner_k, score_fun):
    """Inner loop of a nested cross-validation

    This function estimates the performance of each hyperparameter in
    `C_candidates` with cross validation. It then selects the best
    hyperparameter and refits a model on the whole data using the selected
    hyperparameter. The fitted model is returned.

    Parameters
    ----------
    model : scikit-learn estimator
      The base estimator, copies of which are trained and evaluated. `model`
      itself is not modified.

    C_candidates : list[float]
      list of possible values for the hyperparameter C.

    X : numpy array of shape (n_samples, n_features)
      the design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the target vector

    inner_k : int
      number of cross-validation folds

    score_fun : callable
      the function computing the score on test data, with signature
      `score = score_fun(true_y, predicted_y)`.

    Returns
    -------
    best_model : scikit-learn estimator
      A copy of `model`, fitted on the whole `(X, y)` data, with the
      (estimated) best hyperparameter.

    """
    all_scores = []
    for C in C_candidates:
        print(f"  Grid search: evaluate hyperparameter C = {C}")
        # **TODO** : run 3-fold cross-validation loop, using this particular
        # hyperparameter C. Compute the mean of scores across cross-validation
        # folds and append it to `all_scores`.

    # **TODO**: select the best hyperparameter according to the CV scores,
    # refit the model on the whole data using this hyperparameter, and return
    # the fitted model. Use `model.set_params` to set the hyperparameter
    best_C = "???"
    print(f"  ** Grid search: keep best hyperparameter C = {best_C} **")
    # `clone` is to work with a copy of `model` instead of modifying the
    # argument itself.
    best_model = clone(model)
    # TODO ...
    return best_model


def cross_validate(model, C_candidates, X, y, k, inner_k, score_fun):
    """Get CV score with an inner CV loop to select hyperparameters.

    Parameters
    ----------
    model : scikit-learn estimator, for example `LogisticRegression()`
      The base model to fit and evaluate. `model` itself is not modified.

    C_candidates : list[float]
      list of possible values for the hyperparameter C.

    X : numpy array of shape (n_samples, n_features)
      the design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the target vector

    k : int
      the number of splits for the k-fold cross-validation.

    inner_k : int
      the number of splits for the nested cross-validation (hyperparameter
      selection).

    score_fun : callable
      the function computing the score on test data, with signature
      `score = score_fun(true_y, predicted_y)`.

    Returns
    -------
    scores : list[float]
       The scores obtained for each of the cross-validation folds

    """
    all_scores = []
    for i, (train_idx, test_idx) in enumerate(get_kfold_splits(len(y), k)):
        print(f"\nOuter CV loop: fold {i}")
        best_model = grid_search(
            model, C_candidates, X[train_idx], y[train_idx], inner_k, score_fun
        )
        predictions = best_model.predict(X[test_idx])
        score = score_fun(y[test_idx], predictions)
        print(f"Outer CV loop: finished fold {i}, score: {score:.2f}")
        all_scores.append(score)
    return all_scores


def cross_validate_sklearn(model, C_candidates, X, y, k, inner_k, scoring):
    """CV and hyperparameter selection using scikit-learn.

    This is used as a reference for the output our implementation should
    produce.

    Parameters
    ----------
    model : scikit-learn estimator, for example `LogisticRegression()`
      The base model to fit and evaluate. `model` itself is not modified.

    C_candidates : list[float]
      list of possible values for the hyperparameter C.

    X : numpy array of shape (n_samples, n_features)
      the design matrix

    y : numpy array of shape (n_samples, n_outputs) or (n_samples,)
      the target vector

    k : int
      the number of splits for the k-fold cross-validation.

    inner_k : int
      the number of splits for the nested cross-validation (hyperparameter
      selection).

    scoring : str
      name of a scikit-learn metric

    Returns
    -------
    scores : list[float]
       The scores obtained for each of the cross-validation folds
    """
    grid_search_model = model_selection.GridSearchCV(
        model,
        {"C": C_candidates},
        cv=model_selection.KFold(inner_k),
        scoring=scoring,
    )
    return model_selection.cross_validate(
        grid_search_model, X, y, cv=model_selection.KFold(k), scoring=scoring
    )["test_score"]


# ## Trying our routines on real data
#
# The code below gets executed when you run this script with `python
# nested_cross_validation`. It computes a cross-validation score with our code,
# and compares it to the results obtained with scikit-learn. Read it to see how
# what we have implemented would be easily done with scikit-learn.
#
# Here we have written this logic ourselves to understand how it works, but in
# practice in real projects we would use the scikit-learn functionality which
# is more flexible, more reliable and faster.
#
# Note: this code will only run once you have completed the exercises!


X, y = load_data()
model = linear_model.LogisticRegression()
C_candidates = [0.0001, 0.001, 0.01, 0.1]
k, inner_k = 5, 3

scores = cross_validate(model, C_candidates, X, y, k, inner_k, metrics.accuracy_score)
print("\n\nMy scores:")
print(scores)
sklearn_scores = cross_validate_sklearn(model, C_candidates, X, y, k, inner_k, "accuracy")
print("Scikit-learn scores:")
print(list(sklearn_scores))
assert np.allclose(scores, sklearn_scores), "Results differ from scikit-learn!"

# ## Questions
#
# - When running the cross-validation procedure we have just implemented, how
#   many models did we fit in total?
#   - Answer: 5 * (3 * 4 + 1) = 65
# - There are 150 samples in the iris dataset. For this dataset, what is the
#   size of the 5 test sets in the outer loop? of each of the 3 validation sets
#   in the grid-search (inner loop)?
#   - Answer: outer loop: 150 / 5 = 30; inner loop: (150 * 4 / 5) / 3 = 40
#
# ## Additional exercise (optional)
#
# Have you noticed the hyperparameter grid was specified slightly differently
# for the scikit-learn `GridSearchCV`? we passed a dictionary:
# `{"C": [0.0001, 0.001, 0.01, 0.1 ]}`.
#
# This is because with `GridSearchCV` we can specify values for several
# hyperparameters, for example:
# `{"C": [0.0001, 0.001, 0.01, 0.1], "penalty": ["l1", "l2"]}`,
# and all combinations of these will be tried.
#
# Modify this module so that we can specify such a hyperparameter grid, rather
# than only a list of values for a specific hyperparameter named "C". Hint:
# check the documentation for the `set_params` function of scikit-learn
# estimators. You may want to use the python dict unpacking syntax, for example
# `model.set_params(**hyperparams)`. You can also use `itertools.product` from
# the python standard library to easily build all the combinations of
# hyperparameters.
