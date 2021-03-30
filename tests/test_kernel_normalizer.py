import unittest

import numpy as np
import sklearn

from skcosmo.preprocessing import KernelNormalizer


class KernelTests(unittest.TestCase):
    def test_sample_weights(self):
        """Checks that sample weights of one are equal to the unweighted case and that nonuniform weights are different from the unweighted case"""
        K = np.random.uniform(0, 100, size=(3, 3))
        equal_wts = np.ones(len(K))
        nonequal_wts = np.random.uniform(0, 100, size=(len(K),))
        model = KernelNormalizer()
        weighted_model = KernelNormalizer()
        K_unweighted = model.fit_transform(K)
        K_equal_weighted = weighted_model.fit_transform(K, sample_weight=equal_wts)
        self.assertTrue((np.isclose(K_unweighted, K_equal_weighted, atol=1e-12)).all())
        K_nonequal_weighted = weighted_model.fit_transform(
            K, sample_weight=nonequal_wts
        )
        self.assertFalse(
            (np.isclose(K_unweighted, K_nonequal_weighted, atol=1e-12)).all()
        )

    def test_invalid_sample_weights(self):
        """Checks that weights must be 1D array with the same length as the number of samples"""
        K = np.random.uniform(0, 100, size=(3, 3))
        wts_len = np.ones(len(K) + 1)
        wts_dim = np.ones((len(K), 2))
        model = KernelNormalizer()
        with self.assertRaises(ValueError):
            model.fit_transform(K, sample_weight=wts_len)
        with self.assertRaises(ValueError):
            model.fit_transform(K, sample_weight=wts_dim)

    def test_NoInputs(self):
        """Checks that fit cannot be called with zero inputs."""
        model = KernelNormalizer()
        with self.assertRaises(ValueError):
            model.fit()

    def test_ValueError(self):
        """Checks that a non-square matrix cannot be normalized."""
        K = np.random.uniform(0, 100, size=(3, 4))
        model = KernelNormalizer()
        with self.assertRaises(ValueError):
            model.fit(K)

    def test_reference_ValueError(self):
        """Checks that it is impossible to normalize
        a matrix with a non-coincident size with the reference."""
        K = np.random.uniform(0, 100, size=(3, 3))
        K_2 = np.random.uniform(0, 100, size=(2, 2))
        model = KernelNormalizer()
        model = model.fit(K)
        with self.assertRaises(ValueError):
            model.transform(K_2)

    def test_NotFittedError_transform(self):
        """Checks that an error is returned when
        trying to use the transform function
        before the fit function"""
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelNormalizer()
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            model.transform(K)

    def test_fit_transform(self):
        """Checks that the kernel is correctly normalized.
        Compare with the value calculated
        directly from the equation.
        """
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelNormalizer()
        Ktr = model.fit_transform(K)
        Kc = K - K.mean(axis=0) - K.mean(axis=1)[:, np.newaxis] + K.mean()
        Kc /= np.trace(Kc) / Kc.shape[0]

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_center_only(self):
        """Checks that the kernel is correctly centered,
        but not normalized.
        Compare with the value calculated
        directly from the equation.
        """
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelNormalizer(with_center=True, with_trace=False)
        Ktr = model.fit_transform(K)
        Kc = K - K.mean(axis=0) - K.mean(axis=1)[:, np.newaxis] + K.mean()

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_trace_only(self):
        """Checks that the kernel is correctly normalized,
        but not centered.
        Compare with the value calculated
        directly from the equation.
        """
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelNormalizer(with_center=False, with_trace=True)
        Ktr = model.fit_transform(K)
        Kc = K.copy()
        Kc /= np.trace(Kc) / Kc.shape[0]

        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())

    def test_no_preprocessing(self):
        """Checks that the kernel is unchanged
        if no preprocessing is specified.
        """
        K = np.random.uniform(0, 100, size=(3, 3))
        model = KernelNormalizer(with_center=False, with_trace=False)
        Ktr = model.fit_transform(K)
        Kc = K.copy()
        self.assertTrue((np.isclose(Ktr, Kc, atol=1e-12)).all())


if __name__ == "__main__":
    unittest.main()
