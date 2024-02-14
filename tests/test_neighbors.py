import unittest

import numpy as np

from skmatter.feature_selection import FPS
from skmatter.neighbors import SparseKDE, covariance, effdim, oas


class SparseKDETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(0)
        cls.n_samples_per_cov = 10000
        cls.samples = np.concatenate(
            [
                np.random.multivariate_normal(
                    [0, 0], [[1, 0.5], [0.5, 1]], cls.n_samples_per_cov
                ),
                np.random.multivariate_normal(
                    [4, 4], [[1, 0.5], [0.5, 0.5]], cls.n_samples_per_cov
                ),
            ]
        )
        cls.selector = FPS(n_to_select=int(np.sqrt(2 * cls.n_samples_per_cov)))
        cls.grids = cls.selector.fit_transform(cls.samples.T).T
        cls.expect_weight = np.array([0.49848071, 0.50151929])
        cls.expect_means = np.array(
            [[0.01281471, 0.18859686], [4.22711008, 4.36817619]],
        )
        cls.expect_covs = np.array(
            [
                [[1.10462777, 0.6370178], [0.6370178, 1.11759455]],
                [[1.03559702, 0.50091544], [0.50091544, 0.53316178]],
            ]
        )

        cls.cell = np.array([4, 4])
        cls.expect_weight_periodic = np.array([1.0])
        cls.expect_means_periodic = np.array([[0.01281471, 0.18859686]])
        cls.expect_covs_periodic = np.array([[[0.72231751, 0.0], [0.0, 0.56106493]]])

    def test_sparse_kde(self):
        estimator = SparseKDE(
            self.samples, None, fpoints=0.5, qs=0.85
        )
        estimator.fit(self.grids)
        self.assertTrue(np.allclose(estimator.cluster_weight, self.expect_weight))
        self.assertTrue(np.allclose(estimator.cluster_mean, self.expect_means))
        self.assertTrue(np.allclose(estimator.cluster_cov, self.expect_covs))

    def test_sparse_kde_periodic(self):
        estimator = SparseKDE(
            self.samples,
            None,
            metric_params={"cell": self.cell},
            fpoints=0.5,
            qs=0.85,
        )
        estimator.fit(self.grids)
        self.assertTrue(
            np.allclose(estimator.cluster_weight, self.expect_weight_periodic)
        )
        self.assertTrue(np.allclose(estimator.cluster_mean, self.expect_means_periodic))
        self.assertTrue(np.allclose(estimator.cluster_cov, self.expect_covs_periodic))


class CovarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.array([[1, 2], [3, 3], [4, 6]])
        cls.expected_cov = np.array(
            [[2.33333333, 2.83333333], [2.83333333, 4.33333333]]
        )
        cls.expected_cov_periodic = np.array(
            [[1.12597216, 0.45645371], [0.45645371, 0.82318948]]
        )
        cls.cell = np.array([3, 3])

    def test_covariance(self):
        cov = covariance(self.X, np.full(len(self.X), 1 / len(self.X)), None)
        self.assertTrue(np.allclose(cov, self.expected_cov))

    def test_covariance_periodic(self):
        cov = covariance(self.X, np.full(len(self.X), 1 / len(self.X)), self.cell)
        self.assertTrue(np.allclose(cov, self.expected_cov_periodic))


class EffdimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cov = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        cls.expected_effdim = 1.8898815748423097

    def test_effdim(self):
        self.assertTrue(np.allclose(effdim(self.cov), self.expected_effdim))


class OASTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cov = np.array([[0.5, 1.0], [0.7, 0.4]])
        cls.n = 10
        cls.D = 2
        cls.expected_oas = np.array(
            [[0.48903924, 0.78078484], [0.54654939, 0.41096076]]
        )

    def test_oas(self):
        self.assertTrue(np.allclose(oas(self.cov, self.n, self.D), self.expected_oas))
