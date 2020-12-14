import unittest
from skcosmo.preprocessing.flexible_scaler import KernelFlexibleCenterer
import sklearn
import numpy as np


class KernelTests(unittest.TestCase):
    def test_NoInputs(self):
        """Checks that fit cannot be called with zero inputs."""
        model = KernelFlexibleCenterer()
        with self.assertRaises(AssertionError):
            model.fit()

    def test_ValueError(self):
        """Checks that a non-square matrix cannot be normalized."""
        K = np.random.uniform(0, 100, size=(3, 4))
        model = KernelFlexibleCenterer()
        with self.assertRaises(ValueError):
            model.fit(K)

    def test_reference_ValueError(self):
        """Checks that it is impossible to normalize
        a matrix with a non-coincident size with the reference."""
        K = np.random.uniform(0, 100, size=(3, 3))
        K_2 = np.random.uniform(0, 100, size=(2, 2))
        model = KernelFlexibleCenterer()
        model = model.fit(K)
        with self.assertRaises(ValueError):
            model.transform(K_2)

    def test_NotFittedError_transform(self):
        """Checks that an error is returned when
        trying to use the transform function
        before the fit function"""
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelFlexibleCenterer()
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            model.transform(K)

    def test_fit_transform(self):
        """Checks that the kernel is correctly normalized.
        Compare with the value calculated
        directly from the equation.
        """
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelFlexibleCenterer()
        Ktr = model.fit_transform(K)
        Kc = (
            K
            - np.broadcast_arrays(K, K.mean(axis=0))[1]
            - K.mean(axis=1).reshape((K.shape[0], 1))
            + np.broadcast_arrays(K, K.mean())[1]
        )
        Kc /= np.trace(Kc) / Kc.shape[0]

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_fit_reference(self):
        """Checks that the kernel is correctly normalized
        with reference mean values.
        Compare with the value calculated
        directly from the equation.
        """
        K = np.random.uniform(0, 100, size=(3, 3))
        K_fit_rows = np.random.uniform(0, 100, size=(3))
        K_fit_all = np.random.uniform(0, 100, size=(1))[0]
        model = KernelFlexibleCenterer()
        K_tr = model.fit_transform(K, K_fit_rows=K_fit_rows, K_fit_all=K_fit_all)
        Kc = (
            K
            - np.broadcast_arrays(K, K_fit_rows)[1]
            - K.mean(axis=1).reshape((K.shape[0], 1))
            + np.broadcast_arrays(K, K_fit_all)[1]
        )
        Kc /= np.trace(Kc) / Kc.shape[0]

        self.assertTrue((np.isclose(K_tr, Kc, atol=1e-12)).all())


if __name__ == "__main__":
    unittest.main()
