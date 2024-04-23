## Imports
import argparse

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def extract_connectome_features(func_data, measure):
    """A function to calculate connnectome based on timeseries data and similarity measure"""
    connectome_matrix = measure.fit_transform([func_data])[0]
    tril_idx = np.tril_indices(len(connectome_matrix), k=-1)
    flat_features = connectome_matrix[tril_idx]

    return flat_features


def load_data(n_subjects, parcel, data_dir):
    """Reads data from local directory or nilearn dataset"""
    data = datasets.fetch_abide_pcp(n_subjects=n_subjects, derivatives=[parcel], data_dir=data_dir)
    pheno = pd.DataFrame(data["phenotypic"]).drop(columns=["i", "Unnamed: 0"])

    return data, pheno


def get_train_test_splits(X, y, test_subset_fraction=0.2):
    """Splits samples into a single train-test split"""
    stratification = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,  # input features
        y,  # output labels
        test_size=test_subset_fraction,
        shuffle=True,  # shuffle dataset
        # before splitting
        stratify=stratification,
        random_state=123,  # same shuffle each time
    )

    # print the size of our training and test groups
    print("training:", len(X_train), "testing:", len(X_test))

    return X_train, X_test, y_train, y_test


def run(n_subjects, parcel, data_dir, task, model):
    """Setup and run ML tasks"""
    print("-" * 25)
    print("Loading data")
    print("-" * 25)
    data, pheno = load_data(n_subjects, parcel, data_dir)

    # Imaging variables
    features = data[parcel]
    print(f"Number of samples: {len(features)}")
    subject_feature_shape = features[0].shape
    n_rois = subject_feature_shape[1]
    print(f"subject_feature_shape: {subject_feature_shape}")

    # preprocess fmri data (flatten connectome)
    print("-" * 25)
    print("Flattening the connectome matrix")
    print("-" * 25)
    correlation_measure = ConnectivityMeasure(kind="correlation")

    print(f"Extracting lower triangle values from {n_rois}x{n_rois} connectivity matrix")
    flat_features_list = []
    for func_data in features:
        flat_features = extract_connectome_features(func_data, correlation_measure)
        flat_features_list.append(flat_features)

    # setup X,y for ML model
    print("-" * 25)
    print("Setting up X and y for the ML model")
    print("-" * 25)
    X = np.array(flat_features_list)
    print(f"Input data (X) shape: {X.shape}")

    y = pheno[task]
    y_counts = y.value_counts()

    print(f"Unique output clasess:\n{y_counts}")

    # Encode labels to integer categories
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # Get a single train-test split (80/20)
    X_train, X_test, y_train, y_test = get_train_test_splits(X, y)

    # train model
    if model == "RF":
        clf = RandomForestClassifier(max_depth=3, class_weight="balanced", random_state=0)
    elif model == "LR":
        clf = LogisticRegression(
            penalty="l1", C=1, class_weight="balanced", solver="saga", random_state=0
        )
    else:
        print(f"Unknown model: {model}")

    if model in ["RF", "LR"]:
        print("-" * 25)
        print("Training {model} model")
        print("-" * 25)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        print(f"train acc: {train_acc:.3f}")

        # Evaluate on a test set
        y_pred = clf.predict(X_test)
        test_acc = clf.score(X_test, y_test)
        print(f"test acc: {test_acc:.3f}")

        print("-" * 25)
        print("Other useful performance metrics:")
        print("-" * 25)
        test_cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion matrix:\n{test_cm}")
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

        print(f"precision: {p:.2f}, recall: {r:.2f}, f1: {f1:.2f}")


if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script version of the classification tutorial (diagnosis or scan-site) using ABIDE dataset
    """
    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument(
        "--n_subjects", type=int, default=100, help="number of subjects to download"
    )
    parser.add_argument(
        "--parcel",
        type=str,
        default="rois_ho",
        help="parcellation for connectome (rois_ho or rois_aal)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="data dir for previously downloaded data"
    )
    parser.add_argument(
        "--task", type=str, default="DX_GROUP", help="ML classification task (DX_GROUP or SITE_ID)"
    )
    parser.add_argument("--model", type=str, default="RF", help="ML model to use (RF or LR)")

    args = parser.parse_args()

    n_subjects = args.n_subjects
    parcel = args.parcel
    data_dir = args.data_dir
    task = args.task
    model = args.model

    print("-" * 50)
    print(
        f"Performing {task} classification task using {model} model with {n_subjects} subjects and {parcel} parcellation"
    )
    print("-" * 50)

    run(n_subjects, parcel, data_dir, task, model)

    print("-" * 50)
    print(f"Analysis completed for {task} classification task using {model}!")
    print("-" * 50)
