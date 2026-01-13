import numpy as np
import pytest
import sklearn
from sklearn.preprocessing import StandardScaler

from skmatter.preprocessing import StandardFlexibleScaler


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


def test_sample_weights(random_state):
    """Checks that sample weights of one are equal to the unweighted case.

    Also, that the nonuniform weights are different from the unweighted case
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    equal_wts = np.ones(len(X))
    nonequal_wts = random_state.uniform(0, 100, size=(len(X),))
    model = StandardFlexibleScaler()
    weighted_model = StandardFlexibleScaler()
    X_unweighted = model.fit_transform(X)
    X_equal_weighted = weighted_model.fit_transform(X, sample_weight=equal_wts)
    np.testing.assert_allclose(X_unweighted, X_equal_weighted, atol=1e-12)
    X_nonequal_weighted = weighted_model.fit_transform(X, sample_weight=nonequal_wts)
    assert not np.allclose(X_unweighted, X_nonequal_weighted, atol=1e-12)


def test_invalid_sample_weights(random_state):
    """Checks that weights must be 1D array with the same length as the number of
    samples
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    wts_len = np.ones(len(X) + 1)
    wts_dim = np.ones((len(X), 2))
    model = StandardFlexibleScaler()
    with pytest.raises(ValueError, match="sample_weight.shape"):
        model.fit_transform(X, sample_weight=wts_len)
    with pytest.raises(ValueError, match="Sample weights must be"):
        model.fit_transform(X, sample_weight=wts_dim)


def test_fit_transform_pf(random_state):
    """Checks that in the case of normalization by columns,
    the result is the same as in the case of using the package from sklearn
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler(column_wise=True)
    transformed_skmatter = model.fit_transform(X)
    transformed_sklearn = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(transformed_sklearn, transformed_skmatter, atol=1e-12)


def test_fit_transform_npf(random_state):
    """Checks that the entire matrix is correctly normalized
    (not column-wise). Compare with the value calculated
    directly from the equation.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler(column_wise=False)
    X_tr = model.fit_transform(X)
    mean = X.mean(axis=0)
    var = ((X - mean) ** 2).mean(axis=0)
    scale = np.sqrt(var.sum())
    X_ex = (X - mean) / scale
    np.testing.assert_allclose(X_ex, X_tr, atol=1e-12)


def test_transform(random_state):
    """Checks the transformation relative
    to the reference matrix.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler(column_wise=True)
    model.fit(X)
    Y = random_state.uniform(0, 100, size=(3, 3))
    Y_tr = model.transform(Y)
    mean = X.mean(axis=0)
    var = ((X - mean) ** 2).mean(axis=0)
    scale = np.sqrt(var)
    Y_ex = (Y - mean) / scale
    np.testing.assert_allclose(Y_tr, Y_ex, atol=1e-12)


def test_inverse_transform(random_state):
    """Checks the inverse transformation with
    respect to the reference matrix.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler(column_wise=True)
    model.fit(X)
    Y = random_state.uniform(0, 100, size=(3, 3))
    Y_tr = model.transform(Y)
    Y = np.around(Y, decimals=4)
    Y_inv = np.around((model.inverse_transform(Y_tr)), decimals=4)
    np.testing.assert_allclose(Y, Y_inv, atol=1e-12)
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler(column_wise=False)
    model.fit(X)
    Y = random_state.uniform(0, 100, size=(3, 3))
    Y_tr = model.transform(Y)
    Y = np.around(Y, decimals=4)
    Y_inv = np.around((model.inverse_transform(Y_tr)), decimals=4)
    np.testing.assert_allclose(Y, Y_inv, atol=1e-12)


def test_NotFittedError_transform(random_state):
    """Checks that an error is returned when trying to use the transform function
    before the fit function.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler(column_wise=True)
    match = "instance is not fitted"
    with pytest.raises(sklearn.exceptions.NotFittedError, match=match):
        model.transform(X)


def test_shape_inconsistent_transform(random_state):
    """Checks that an error is returned when attempting to use the transform
    function with mismatched matrix sizes.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    X_test = random_state.uniform(0, 100, size=(4, 4))
    model = StandardFlexibleScaler(column_wise=True)
    model.fit(X)
    with pytest.raises(ValueError):
        model.transform(X_test)


def test_shape_inconsistent_inverse(random_state):
    """Checks that an error is returned when attempting to use the inverse transform
    function with mismatched matrix sizes.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    X_test = random_state.uniform(0, 100, size=(4, 4))
    model = StandardFlexibleScaler(column_wise=True)
    model.fit(X)
    with pytest.raises(ValueError):
        model.inverse_transform(X_test)


def test_NotFittedError_inverse(random_state):
    """Checks that an error is returned when trying to use the inverse transform
    function before the fit function.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    model = StandardFlexibleScaler()
    with pytest.raises(sklearn.exceptions.NotFittedError):
        model.inverse_transform(X)


def test_ValueError_column_wise(random_state):
    """Checks that the matrix cannot be normalized across columns if there is a zero
    variation column.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    X[0][0] = X[1][0] = X[2][0] = 2
    model = StandardFlexibleScaler(column_wise=True)
    with pytest.raises(ValueError):
        model.fit(X)


def test_atol(random_state):
    """Checks that we can define absolute tolerance and it control the
    minimal variance of columns ot the whole matrix.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    atol = ((X[:, 0] - X[:, 0].mean(axis=0)) ** 2).mean(axis=0) + 1e-8
    model = StandardFlexibleScaler(column_wise=True, atol=atol, rtol=0)
    with pytest.raises(ValueError):
        model.fit(X)
    atol = (X - X.mean(axis=0) ** 2).mean(axis=0) + 1e-8
    model = StandardFlexibleScaler(column_wise=False, atol=atol, rtol=0)
    with pytest.raises(ValueError):
        model.fit(X)


def test_rtol(random_state):
    """Checks that we can define relative tolerance and it control the
    minimal variance of columns or the whole matrix.
    """
    X = random_state.uniform(0, 100, size=(3, 3))
    mean = X[:, 0].mean(axis=0)
    rtol = ((X[:, 0] - mean) ** 2).mean(axis=0) / mean + 1e-8
    model = StandardFlexibleScaler(column_wise=True, atol=0, rtol=rtol)
    with pytest.raises(ValueError):
        model.fit(X)
    mean = X.mean(axis=0)
    rtol = ((X - mean) ** 2).mean(axis=0) / mean + 1e-8
    model = StandardFlexibleScaler(column_wise=False, atol=0, rtol=rtol)
    with pytest.raises(ValueError):
        model.fit(X)


def test_ValueError_full(random_state):
    """Checks that the matrix cannot be normalized if there is a zero variation
    matrix.
    """
    X = np.array([2, 2, 2]).reshape(-1, 1)
    model = StandardFlexibleScaler(column_wise=False)
    with pytest.raises(ValueError):
        model.fit(X)


def test_not_w_mean(random_state):
    """Checks that the matrix normalized `with_mean=False` does not have a mean."""
    X = np.array([2, 2, 3]).reshape(-1, 1)
    model = StandardFlexibleScaler(with_mean=False)
    model.fit(X)
    np.testing.assert_allclose(model.mean_, 0)
