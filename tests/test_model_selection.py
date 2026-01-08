import pytest
import sklearn.model_selection
from sklearn.datasets import load_iris

import skmatter.model_selection


@pytest.fixture(scope="module")
def test_data():
    X = load_iris().data[:10]
    seed = 0x5F3759DF
    return {"X": X, "seed": seed}


def test_train_test_splits(test_data):
    # see if train_test_split of skmatter agrees with the one of sklearn
    X = test_data["X"]
    seed = test_data["seed"]
    sklearn_outputs = sklearn.model_selection.train_test_split(X, random_state=seed)
    skmatter_outputs = skmatter.model_selection.train_test_split(X, random_state=seed)
    for i in range(len(skmatter_outputs)):
        assert (sklearn_outputs[i] == skmatter_outputs[i]).all()


def test_train_test_splits_train_test_overlap(test_data):
    # tests that a test/train split which necessitates overlap returns the right
    # number of points in each set
    X = test_data["X"]
    seed = test_data["seed"]
    X_train, X_test = skmatter.model_selection.train_test_split(
        X,
        train_size=0.8,
        test_size=0.8,
        train_test_overlap=True,
        random_state=seed,
    )
    assert len(X_train) == len(X_test) == int(0.8 * X.shape[0])


def test_train_test_splits_train_test_overlap_full_test_set(test_data):
    # tests that the entire dataset can be used as the testing set
    X = test_data["X"]
    seed = test_data["seed"]
    X_train, X_test = skmatter.model_selection.train_test_split(
        X,
        train_size=0.8,
        test_size=1.0,
        train_test_overlap=True,
        random_state=seed,
    )
    assert (X == X_test).all()


def test_train_test_splits_train_test_overlap_full_train_test_set(test_data):
    # tests that the full dataset can be "split" to both train and test set
    X = test_data["X"]
    seed = test_data["seed"]
    X_train, X_test = skmatter.model_selection.train_test_split(
        X,
        train_size=1.0,
        test_size=1.0,
        train_test_overlap=True,
        random_state=seed,
    )
    assert (X_train == X_test).all()
