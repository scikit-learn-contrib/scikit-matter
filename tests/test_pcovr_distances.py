import unittest

import numpy as np
import scipy
from sklearn.datasets import load_boston

from skcosmo.utils import (
    pcovr_covariance,
    pcovr_kernel,
)


class CovarianceTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.Y = load_boston(return_X_y=True)

    def test_alphas(self):
        C_X = self.X.T @ self.X

        C_inv = np.linalg.pinv(C_X, rcond=1e-12)
        C_isqrt = np.real(scipy.linalg.sqrtm(C_inv))

        # parentheses speed up calculation greatly
        C_Y = C_isqrt @ (self.X.T @ self.Y)
        C_Y = C_Y.reshape((C_X.shape[0], -1))
        C_Y = np.real(C_Y)
        C_Y = C_Y @ C_Y.T

        for alpha in [0.0, 0.5, 1.0]:
            with self.subTest(alpha=alpha):
                C = pcovr_covariance(alpha, X=self.X, Y=self.Y, rcond=1e-6)
                self.assertTrue(np.allclose(C, alpha * C_X + (1 - alpha) * C_Y))

    def test_no_return_isqrt(self):
        with self.assertRaises(ValueError):
            _, _ = pcovr_covariance(0.5, self.X, self.Y, return_isqrt=False)

    def test_inverse_covariance(self):
        rcond = 1e-12
        rng = np.random.default_rng(0)

        # Make some random data where the last feature
        # is a linear comibination of the other features.
        # This gives us a covariance with a zero eigenvalue
        # that should be dropped (via rcond).
        # Hence, the inverse square root covariance
        # should be identical between the "full"
        # computation (eigh) and the approximate
        # computation that takes the top n_features-1
        # singular values (randomized svd).
        X = rng.random((10, 5))
        Y = rng.random(10)
        x = rng.random(5)
        Xx = np.column_stack((X, np.sum(X * x, axis=1)))
        Xx -= np.mean(Xx, axis=0)

        C_inv = np.linalg.pinv(Xx.T @ Xx, rcond=rcond)
        C_isqrt = np.real(scipy.linalg.sqrtm(C_inv))

        X_shape = min(self.X.shape)
        _, C_isqrt_eigh = pcovr_covariance(0.5, Xx, Y, return_isqrt=True, rcond=rcond)
        _, C_isqrt_svd = pcovr_covariance(
            0.5, Xx, Y, return_isqrt=True, rank=X_shape - 1, rcond=rcond
        )

        for C, C_type in zip([C_isqrt_eigh, C_isqrt_svd], ["eigh", "svd"]):
            with self.subTest(C_isqrt_type=C_type):
                self.assertTrue(np.allclose(C_isqrt, C))


class KernelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.Y = load_boston(return_X_y=True)

    def test_alphas(self):
        K_X = self.X @ self.X.T
        K_Y = self.Y @ self.Y.T

        for alpha in [0.0, 0.5, 1.0]:
            with self.subTest(alpha=alpha):
                K = pcovr_kernel(alpha, self.X, self.Y)
                self.assertTrue(np.allclose(K, alpha * K_X + (1 - alpha) * K_Y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
