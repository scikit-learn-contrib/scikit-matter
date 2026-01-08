import numpy as np
import pytest
import scipy
from sklearn.datasets import load_diabetes as get_dataset

from skmatter.utils import pcovr_covariance, pcovr_kernel


@pytest.fixture(scope="module")
def dataset():
    return get_dataset(return_X_y=True)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_covariance_alphas(dataset, alpha):
    X, Y = dataset
    C_X = X.T @ X

    C_inv = np.linalg.pinv(C_X, rcond=1e-12)
    C_isqrt = np.real(scipy.linalg.sqrtm(C_inv))

    # parentheses speed up calculation greatly
    C_Y = C_isqrt @ (X.T @ Y)
    C_Y = C_Y.reshape((C_X.shape[0], -1))
    C_Y = np.real(C_Y)
    C_Y = C_Y @ C_Y.T

    C = pcovr_covariance(alpha, X=X, Y=Y, rcond=1e-6)
    np.testing.assert_allclose(C, alpha * C_X + (1 - alpha) * C_Y)


def test_no_return_isqrt(dataset):
    X, Y = dataset
    with pytest.raises(ValueError, match="too many values to unpack"):
        _, _ = pcovr_covariance(0.5, X, Y, return_isqrt=False)


@pytest.mark.parametrize("C_isqrt_type", ["eigh", "svd"])
def test_inverse_covariance(C_isqrt_type):
    rcond = 1e-12
    rng = np.random.default_rng(0)

    # Make some random data where the last feature is a linear comibination of the other
    # features. This gives us a covariance with a zero eigenvalue that should be dropped
    # (via rcond). Hence, the inverse square root covariance should be identical between
    # the "full" computation (eigh) and the approximate computation that takes the top
    # n_features-1 singular values (randomized svd).

    X = rng.random((10, 5))
    Y = rng.random(10)
    x = rng.random(5)
    Xx = np.column_stack((X, np.sum(X * x, axis=1)))
    Xx -= np.mean(Xx, axis=0)

    C_inv = np.linalg.pinv(Xx.T @ Xx, rcond=rcond)
    C_isqrt = np.real(scipy.linalg.sqrtm(C_inv))

    if C_isqrt_type == "eigh":
        _, C_isqrt_computed = pcovr_covariance(
            0.5, Xx, Y, return_isqrt=True, rcond=rcond
        )
    else:  # svd
        _, C_isqrt_computed = pcovr_covariance(
            0.5, Xx, Y, return_isqrt=True, rank=min(Xx.shape) - 1, rcond=rcond
        )

    np.testing.assert_allclose(C_isqrt, C_isqrt_computed)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_kernel_alphas(dataset, alpha):
    X, Y = dataset
    K_X = X @ X.T
    K_Y = Y @ Y.T

    K = pcovr_kernel(alpha, X, Y)
    np.testing.assert_allclose(K, alpha * K_X + (1 - alpha) * K_Y)
