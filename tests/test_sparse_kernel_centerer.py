import unittest

import numpy as np
import sklearn

from skcosmo.preprocessing import SparseKernelCenterer


class SparseKernelTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(0)

    def test_sample_weights(self):
        """Checks that sample weights of one are equal to the unweighted case and that the nonuniform weights are different from the unweighted case"""
        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        equal_wts = np.ones(len(Knm))
        nonequal_wts = self.random_state.uniform(-1, 1, size=(len(Knm),))
        model = SparseKernelCenterer()
        weighted_model = SparseKernelCenterer()
        Knm_unweighted = model.fit_transform(Knm, Kmm)
        Knm_equal_weighted = weighted_model.fit_transform(
            Knm, Kmm, sample_weight=equal_wts
        )
        Knm_nonequal_weighted = weighted_model.fit_transform(
            Knm, Kmm, sample_weight=nonequal_wts
        )
        self.assertTrue(
            (np.isclose(Knm_unweighted, Knm_equal_weighted, atol=1e-12)).all()
        )
        self.assertFalse(
            (np.isclose(Knm_unweighted, Knm_nonequal_weighted, atol=1e-12)).all()
        )

    def test_invalid_sample_weights(self):
        """Checks that weights must be 1D array with the same length as the number of samples"""
        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        wts_len = np.ones(len(Knm) + 1)
        wts_dim = np.ones((len(Knm), 2))
        model = SparseKernelCenterer()
        with self.assertRaises(ValueError):
            model.fit_transform(Knm, Kmm, sample_weight=wts_len)
        with self.assertRaises(ValueError):
            model.fit_transform(Knm, Kmm, sample_weight=wts_dim)

    def test_Square_Kmm(self):
        """Checks that the passed active kernel is square"""

        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X.T

        model = SparseKernelCenterer()
        with self.assertRaises(ValueError) as cm:
            model.fit(Knm, Kmm)
            self.assertEqual(str(cm.exception), "The active kernel is not square.")

    def test_LatterDim(self):
        """Checks that a matrix must have the same latter dimension as its active counterpart cannot be normalized."""

        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

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
        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

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
        K = self.random_state.uniform(0, 100, size=(3, 3))
        model = SparseKernelCenterer()
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            model.transform(K)

    def test_fit_transform(self):
        """Checks that the kernel is correctly normalized.
        Compare with the value calculated
        directly from the equation.
        """

        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        model = SparseKernelCenterer(rcond=1e-12)
        Ktr = model.fit_transform(Knm, Kmm)

        Knm_mean = Knm.mean(axis=0)

        Kc = Knm - Knm_mean

        Khat = Kc @ np.linalg.pinv(Kmm, rcond=1e-12) @ Kc.T

        Kc /= np.sqrt(np.trace(Khat) / Khat.shape[0])

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_center_only(self):
        """Checks that the kernel is correctly centered, but not normalized.
        Compare with the value calculated
        directly from the equation.
        """
        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        model = SparseKernelCenterer(with_center=True, with_trace=False, rcond=1e-12)
        Ktr = model.fit_transform(Knm, Kmm)

        Knm_mean = Knm.mean(axis=0)

        Kc = Knm - Knm_mean

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_trace_only(self):
        """Checks that the kernel is correctly normalized, but not centered.
        Compare with the value calculated
        directly from the equation.
        """
        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        model = SparseKernelCenterer(with_center=False, with_trace=True, rcond=1e-12)
        Ktr = model.fit_transform(Knm, Kmm)

        Kc = Knm.copy()

        Khat = Kc @ np.linalg.pinv(Kmm, rcond=1e-12) @ Kc.T

        Kc /= np.sqrt(np.trace(Khat) / Khat.shape[0])

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_no_preprocessing(self):
        """Checks that the kernel is unchanged
        if no preprocessing is specified.
        """
        X = self.random_state.uniform(-1, 1, size=(4, 5))
        X_sparse = self.random_state.uniform(-1, 1, size=(3, 5))

        Knm = X @ X_sparse.T
        Kmm = X_sparse @ X_sparse.T

        model = SparseKernelCenterer(with_center=False, with_trace=False, rcond=1e-12)
        Ktr = model.fit_transform(Knm, Kmm)

        Kc = Knm.copy()

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())


if __name__ == "__main__":
    unittest.main()
