import numpy as np
import pytest
import sklearn

from skmatter.preprocessing import SparseKernelCenterer


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


def test_sample_weights(random_state):
    """Checks that sample weights of one are equal to the unweighted case and that
    the nonuniform weights are different from the unweighted case.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    equal_wts = np.ones(len(Knm))
    nonequal_wts = random_state.uniform(-1, 1, size=(len(Knm),))
    model = SparseKernelCenterer()
    weighted_model = SparseKernelCenterer()
    Knm_unweighted = model.fit_transform(Knm, Kmm)
    Knm_equal_weighted = weighted_model.fit_transform(Knm, Kmm, sample_weight=equal_wts)
    Knm_nonequal_weighted = weighted_model.fit_transform(
        Knm, Kmm, sample_weight=nonequal_wts
    )
    np.testing.assert_allclose(Knm_unweighted, Knm_equal_weighted, atol=1e-12)
    assert not np.allclose(Knm_unweighted, Knm_nonequal_weighted, atol=1e-12)


def test_invalid_sample_weights(random_state):
    """Checks that weights must be 1D array with the same length as the number of
    samples.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    wts_len = np.ones(len(Knm) + 1)
    wts_dim = np.ones((len(Knm), 2))
    model = SparseKernelCenterer()
    with pytest.raises(ValueError, match="sample_weight.shape"):
        model.fit_transform(Knm, Kmm, sample_weight=wts_len)
    with pytest.raises(ValueError, match="Sample weights must be"):
        model.fit_transform(Knm, Kmm, sample_weight=wts_dim)


def test_Square_Kmm(random_state):
    """Checks that the passed active kernel is square."""
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X.T

    model = SparseKernelCenterer()
    with pytest.raises(ValueError, match="The active kernel is not square."):
        model.fit(Knm, Kmm)


def test_LatterDim(random_state):
    """Checks that a matrix must have the same latter dimension as its active
    counterpart cannot be normalized.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X.T
    Kmm = X_sparse @ X_sparse.T

    model = SparseKernelCenterer()
    match = "The reference kernel is not commensurate shape with the active kernel."
    with pytest.raises(ValueError, match=match):
        model.fit(Knm, Kmm)


def test_new_kernel(random_state):
    """Checks that it is impossible to normalize a matrix with a non-coincident size
    with the reference.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    Knm2 = X @ X.T
    model = SparseKernelCenterer()
    model = model.fit(Knm, Kmm)
    match = "The reference kernel and received kernel have different shape"
    with pytest.raises(ValueError, match=match):
        model.transform(Knm2)


def test_NotFittedError_transform(random_state):
    """Checks that an error is returned when trying to use the transform function
    before the fit function
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    model = SparseKernelCenterer()
    match = "instance is not fitted"
    with pytest.raises(sklearn.exceptions.NotFittedError, match=match):
        model.transform(K)


def test_fit_transform(random_state):
    """Checks that the kernel is correctly normalized.

    Compare with the value calculated directly from the equation.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    model = SparseKernelCenterer(rcond=1e-12)
    Ktr = model.fit_transform(Knm, Kmm)

    Knm_mean = Knm.mean(axis=0)

    Kc = Knm - Knm_mean

    Khat = Kc @ np.linalg.pinv(Kmm, rcond=1e-12) @ Kc.T

    Kc /= np.sqrt(np.trace(Khat) / Khat.shape[0])

    np.testing.assert_allclose(Ktr, Kc, atol=1e-12)


def test_center_only(random_state):
    """Checks that the kernel is correctly centered, but not normalized.
    Compare with the value calculated
    directly from the equation.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    model = SparseKernelCenterer(with_center=True, with_trace=False, rcond=1e-12)
    Ktr = model.fit_transform(Knm, Kmm)

    Knm_mean = Knm.mean(axis=0)

    Kc = Knm - Knm_mean

    np.testing.assert_allclose(Ktr, Kc, atol=1e-12)


def test_trace_only(random_state):
    """Checks that the kernel is correctly normalized, but not centered.
    Compare with the value calculated
    directly from the equation.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    model = SparseKernelCenterer(with_center=False, with_trace=True, rcond=1e-12)
    Ktr = model.fit_transform(Knm, Kmm)

    Kc = Knm.copy()

    Khat = Kc @ np.linalg.pinv(Kmm, rcond=1e-12) @ Kc.T

    Kc /= np.sqrt(np.trace(Khat) / Khat.shape[0])

    np.testing.assert_allclose(Ktr, Kc, atol=1e-12)


def test_no_preprocessing(random_state):
    """Checks that the kernel is unchanged
    if no preprocessing is specified.
    """
    X = random_state.uniform(-1, 1, size=(4, 5))
    X_sparse = random_state.uniform(-1, 1, size=(3, 5))

    Knm = X @ X_sparse.T
    Kmm = X_sparse @ X_sparse.T

    model = SparseKernelCenterer(with_center=False, with_trace=False, rcond=1e-12)
    Ktr = model.fit_transform(Knm, Kmm)

    Kc = Knm.copy()

    np.testing.assert_allclose(Ktr, Kc, atol=1e-12)
