import numpy as np
import pytest
import sklearn

from skmatter.preprocessing import KernelNormalizer


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


def test_sample_weights(random_state):
    """Checks that sample weights of one are equal to the unweighted case and
    that nonuniform weights are different from the unweighted case.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    equal_wts = np.ones(len(K))
    nonequal_wts = random_state.uniform(0, 100, size=(len(K),))
    model = KernelNormalizer()
    weighted_model = KernelNormalizer()
    K_unweighted = model.fit_transform(K)
    K_equal_weighted = weighted_model.fit_transform(K, sample_weight=equal_wts)
    assert (np.isclose(K_unweighted, K_equal_weighted, atol=1e-12)).all()
    K_nonequal_weighted = weighted_model.fit_transform(K, sample_weight=nonequal_wts)
    assert not (np.isclose(K_unweighted, K_nonequal_weighted, atol=1e-12)).all()


def test_invalid_sample_weights(random_state):
    """Checks that weights must be 1D array with the same length as the number of
    samples.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    wts_len = np.ones(len(K) + 1)
    wts_dim = np.ones((len(K), 2))
    model = KernelNormalizer()
    with pytest.raises(ValueError):
        model.fit_transform(K, sample_weight=wts_len)
    with pytest.raises(ValueError):
        model.fit_transform(K, sample_weight=wts_dim)


def test_ValueError(random_state):
    """Checks that a non-square matrix cannot be normalized."""
    K = random_state.uniform(0, 100, size=(3, 4))
    model = KernelNormalizer()
    with pytest.raises(ValueError):
        model.fit(K)


def test_reference_ValueError(random_state):
    """Checks that it is impossible to normalize a matrix with a non-coincident
    size with the reference.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    K_2 = random_state.uniform(0, 100, size=(2, 2))
    model = KernelNormalizer()
    model = model.fit(K)
    with pytest.raises(ValueError):
        model.transform(K_2)


def test_NotFittedError_transform(random_state):
    """Checks that an error is returned when trying to use the transform function
    before the fit function.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    model = KernelNormalizer()
    with pytest.raises(sklearn.exceptions.NotFittedError):
        model.transform(K)


def test_fit_transform(random_state):
    """Checks that the kernel is correctly normalized.

    Compare with the value calculated directly from the equation.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    model = KernelNormalizer()
    Ktr = model.fit_transform(K)
    Kc = K - K.mean(axis=0) - K.mean(axis=1)[:, np.newaxis] + K.mean()
    Kc /= np.trace(Kc) / Kc.shape[0]

    assert (np.isclose(Ktr, Kc, atol=1e-12)).all()


def test_center_only(random_state):
    """Checks that the kernel is correctly centered,
    but not normalized.
    Compare with the value calculated
    directly from the equation.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    model = KernelNormalizer(with_center=True, with_trace=False)
    Ktr = model.fit_transform(K)
    Kc = K - K.mean(axis=0) - K.mean(axis=1)[:, np.newaxis] + K.mean()

    assert (np.isclose(Ktr, Kc, atol=1e-12)).all()


def test_trace_only(random_state):
    """Checks that the kernel is correctly normalized,
    but not centered.
    Compare with the value calculated
    directly from the equation.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    model = KernelNormalizer(with_center=False, with_trace=True)
    Ktr = model.fit_transform(K)
    Kc = K.copy()
    Kc /= np.trace(Kc) / Kc.shape[0]

    assert (np.isclose(Ktr, Kc, atol=1e-12)).all()


def test_no_preprocessing(random_state):
    """Checks that the kernel is unchanged
    if no preprocessing is specified.
    """
    K = random_state.uniform(0, 100, size=(3, 3))
    model = KernelNormalizer(with_center=False, with_trace=False)
    Ktr = model.fit_transform(K)
    Kc = K.copy()
    assert (np.isclose(Ktr, Kc, atol=1e-12)).all()
