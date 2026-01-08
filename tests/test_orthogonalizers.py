import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from skmatter.datasets import load_csd_1000r
from skmatter.utils import (
    X_orthogonalizer,
    Y_feature_orthogonalizer,
    Y_sample_orthogonalizer,
)


EPSILON = 1e-8


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


@pytest.fixture
def n_samples():
    return 2


@pytest.fixture
def n_features():
    return 4


@pytest.mark.parametrize("test_type", ["random X", "correlated X"])
def test_null_column(random_state, n_samples, n_features, test_type):
    # checks that the column passed to the orthogonalizer
    # is empty post-orthogonalization

    n_uncorrelated = n_features // 2

    X_random = random_state.uniform(-1, 1, size=(n_samples, n_features))
    X_correlated = np.zeros((n_samples, n_features))
    X_correlated[:, :n_uncorrelated] = random_state.uniform(
        -1, 1, size=(n_samples, n_uncorrelated)
    )

    for i in range(n_uncorrelated, n_features):
        X_correlated[:, i] = X_correlated[:, i - n_uncorrelated] * random_state.uniform(
            -1, 1
        )

    feat_idx = np.arange(min(n_samples, n_features, n_uncorrelated))
    random_state.shuffle(feat_idx)

    for idx in feat_idx:
        if test_type == "random X":
            X_random = X_orthogonalizer(X_random, c=idx)
            assert np.linalg.norm(X_random[:, idx]) <= EPSILON
        else:  # correlated X
            X_correlated = X_orthogonalizer(X_correlated, c=idx)
            assert np.linalg.norm(X_correlated[:, idx]) <= EPSILON
            assert np.linalg.norm(X_correlated[:, idx + n_uncorrelated]) <= EPSILON


@pytest.mark.parametrize(
    "test_type", ["random X", "random X with column", "correlated X"]
)
def test_null_row(random_state, n_samples, n_features, test_type):
    # checks that the row passed to the orthogonalizer
    # is empty post-orthogonalization

    n_uncorrelated = n_samples // 2

    X_random = random_state.uniform(-1, 1, size=(n_samples, n_features))
    X_random2 = random_state.uniform(-1, 1, size=(n_samples, n_features))
    X_correlated = np.zeros((n_samples, n_features))
    X_correlated[:n_uncorrelated] = random_state.uniform(
        -1, 1, size=(n_uncorrelated, n_features)
    )

    for i in range(n_uncorrelated, n_samples):
        X_correlated[i] = X_correlated[i - n_uncorrelated] * random_state.uniform(-1, 1)

    feat_idx = np.arange(min(n_samples, n_features, n_uncorrelated))
    random_state.shuffle(feat_idx)

    for idx in feat_idx:
        if test_type == "random X":
            X_random = X_orthogonalizer(X_random.T, c=idx).T
            assert np.linalg.norm(X_random[idx]) <= EPSILON
        elif test_type == "random X with column":
            X_random2 = X_orthogonalizer(X_random2.T, x2=X_random2[idx].T).T
            assert np.linalg.norm(X_random2[idx]) <= EPSILON
        else:  # correlated X
            X_correlated = X_orthogonalizer(X_correlated.T, c=idx).T
            assert np.linalg.norm(X_correlated[idx]) <= EPSILON
            assert np.linalg.norm(X_correlated[idx + n_uncorrelated]) <= EPSILON


def test_multiple_orthogonalizations(random_state, n_samples, n_features):
    # checks that the matrix is empty when orthogonalized simultaneously
    # by all uncorrelated columns

    n_uncorrelated = n_samples // 2

    X_correlated = np.zeros((n_samples, n_features))
    X_correlated[:, :n_uncorrelated] = random_state.uniform(
        -1, 1, size=(n_samples, n_uncorrelated)
    )

    for i in range(n_uncorrelated, n_features):
        X_correlated[:, i] = X_correlated[:, i - n_uncorrelated] * random_state.uniform(
            -1, 1
        )

    X_correlated = X_orthogonalizer(X_correlated, x2=X_correlated[:, :n_uncorrelated])
    print(X_correlated)

    assert np.linalg.norm(X_correlated) <= EPSILON


def test_multicolumn(random_state, n_samples, n_features):
    # checks that an error is raised when x2 is the wrong shape for x1
    with pytest.raises(ValueError) as cm:
        X_orthogonalizer(
            random_state.uniform(-3, 3, size=(n_samples, n_features)),
            x2=random_state.uniform(-3, 3, size=(n_samples + 4, n_features)),
        )
    assert str(cm.value) == (
        "You can only orthogonalize a matrix using a vector with the same number "
        f"of rows. Matrix X has {n_samples} rows, whereas the "
        f"orthogonalizing matrix has {n_samples + 4} rows."
    )


def test_warning(n_samples, n_features):
    # checks that a warning is raised when trying to orthogonalize by
    # an empty vector
    with pytest.warns(UserWarning, match="Column vector contains only zeros."):
        X_orthogonalizer(np.zeros((n_samples, n_features)), 0)


def test_copy(random_state, n_samples, n_features):
    # checks that the X_orthogonalizer works in-place when copy=False

    X_random = random_state.uniform(-1, 1, size=(n_samples, n_features))

    idx = random_state.choice(X_random.shape[-1])

    new_X = X_orthogonalizer(X_random, idx, tol=EPSILON, copy=True)
    X_orthogonalizer(X_random, idx, tol=EPSILON, copy=False)
    np.testing.assert_allclose(X_random, new_X)


@pytest.fixture(scope="module")
def csd_data():
    X, y = load_csd_1000r(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y)
    return X, y


def test_pass_feature(csd_data):
    # checks that the Y_feature_orthogonalizer removes all targets
    # predictable by the given set of features
    random_state = np.random.RandomState(0)
    X, y = csd_data

    Xc = X[:, random_state.choice(X.shape[-1], 3)]
    yhat = Xc @ np.linalg.pinv(Xc.T @ Xc, rcond=EPSILON) @ Xc.T @ y

    new_y = Y_feature_orthogonalizer(y, Xc, tol=EPSILON)
    np.testing.assert_allclose(y - new_y, yhat)


def test_copy_feature(csd_data):
    # checks the Y_feature_orthogonalizer operates in-place when copy=False
    random_state = np.random.RandomState(0)
    X, y = csd_data

    Xc = X[:, random_state.choice(X.shape[-1], 3)]
    new_y = Y_feature_orthogonalizer(y, Xc, tol=EPSILON, copy=False)
    np.testing.assert_allclose(y, new_y)


def test_pass_sample(csd_data):
    # checks that the Y_samples_orthogonalizer removes all targets
    # predictable by the given set of samples
    random_state = np.random.RandomState(0)
    X, y = csd_data

    r = random_state.choice(X.shape[0], 3)
    Xr = X[r]
    yr = y[r]

    yhat = X @ np.linalg.pinv(Xr.T @ Xr, rcond=EPSILON) @ Xr.T @ yr

    new_y = Y_sample_orthogonalizer(y, X, yr, Xr, tol=EPSILON)
    np.testing.assert_allclose(y - new_y, yhat)


def test_copy_sample(csd_data):
    # checks the Y_sample_orthogonalizer operates in-place when copy=False
    random_state = np.random.RandomState(0)
    X, y = csd_data

    r = random_state.choice(X.shape[0], 3)
    Xr = X[r]
    yr = y[r]

    new_y = Y_sample_orthogonalizer(y, X, yr, Xr, tol=EPSILON, copy=False)
    np.testing.assert_allclose(y, new_y)
