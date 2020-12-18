import unittest
from skcosmo.preprocessing.flexible_scaler import SparseKernelCenterer
import sklearn
import numpy as np


class SparseKernelTests(unittest.TestCase):
    def test_Square_Kmm(self):
        """Checks that the passed active kernel is square"""

        X = np.random.uniform(-1, 1, size=(4, 5))
        X_sparse = np.random.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X.T

        model = SparseKernelCenterer()
        with self.assertRaises(ValueError) as cm:
            model.fit(Knm, Kmm)
            self.assertEqual(str(cm.exception), "The active kernel is not square.")

    def test_LatterDim(self):
        """Checks that a matrix must have the same latter dimension as its active counterpart cannot be normalized."""

        X = np.random.uniform(-1, 1, size=(4, 5))
        X_sparse = np.random.uniform(-1, 1, size=(3, 5))

        Knm = X @ X.T
        Kmm = X_sparse @ X_sparse.T

        model = SparseKernelCenterer()
        with self.assertRaises(ValueError) as cm:
            model.fit(Knm, Kmm)
            self.assertEqual(
                str(cm.exception),
                "The reference kernel is not "
                "commensurate shape with the active kernel.",
            )

    def test_new_kernel(self):
        """Checks that it is impossible to normalize
        a matrix with a non-coincident size with the reference."""
        X = np.random.uniform(-1, 1, size=(4, 5))
        X_sparse = np.random.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        Knm2 = X @ X.T
        model = SparseKernelCenterer()
        model = model.fit(Knm, Kmm)
        with self.assertRaises(ValueError) as cm:
            model.transform(Knm2)
            self.assertEquals(
                str(cm.exception),
                "The reference kernel and received kernel have different shape",
            )

    def test_NotFittedError_transform(self):
        """Checks that an error is returned when
        trying to use the transform function
        before the fit function"""
        K = np.random.uniform(0, 100, size=(3, 3))
        model = SparseKernelCenterer()
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            model.transform(K)

    def test_fit_transform(self):
        """Checks that the kernel is correctly normalized.
        Compare with the value calculated
        directly from the equation.
        """

        X = np.random.uniform(-1, 1, size=(4, 5))
        X_sparse = np.random.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        model = SparseKernelCenterer(rcond=1e-12)
        Ktr = model.fit_transform(Knm, Kmm)

        Knm_mean = Knm.mean(axis=0)

        Kc = Knm - np.broadcast_arrays(Knm, Knm_mean)[1]

        Khat = Kc @ np.linalg.pinv(Kmm, rcond=1e-12) @ Kc.T

        Kc /= np.sqrt(np.trace(Khat) / Khat.shape[0])

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())


if __name__ == "__main__":
    unittest.main()
