import unittest

import numpy as np

from skmatter.feature_selection import FPS
from skmatter.neighbors import SparseKDE
from skmatter.neighbors._sparsekde import _covariance
from skmatter.utils import effdim, oas


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
        cls.expect_score = -759.831

        cls.cell = np.array([4, 4])
        cls.expect_score_periodic = -456.744

    def test_sparse_kde(self):
        estimator = SparseKDE(self.samples, None, fpoints=0.5)
        estimator.fit(self.grids)
        self.assertTrue(round(estimator.score(self.grids), 3) == self.expect_score)

    def test_sparse_kde_periodic(self):
        estimator = SparseKDE(
            self.samples,
            None,
            metric_params={"cell": self.cell},
            fpoints=0.5,
        )
        estimator.fit(self.grids)
        self.assertTrue(
            round(estimator.score(self.grids), 3) == self.expect_score_periodic
        )

    def test_dimension_check(self):
        estimator = SparseKDE(
            self.samples, None, metric_params={"cell": self.cell}, fpoints=0.5
        )
        self.assertRaises(ValueError, estimator.fit, np.array([[4]]))


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
        cov = _covariance(self.X, np.full(len(self.X), 1 / len(self.X)), None)
        self.assertTrue(np.allclose(cov, self.expected_cov))

    def test_covariance_periodic(self):
        cov = _covariance(self.X, np.full(len(self.X), 1 / len(self.X)), self.cell)
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
