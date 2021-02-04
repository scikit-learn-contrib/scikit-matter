import unittest
from skcosmo.utils import pcovr_covariance, pcovr_kernel
from sklearn.datasets import load_boston
import numpy as np
import scipy


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
