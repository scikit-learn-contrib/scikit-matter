import unittest

import numpy as np

from skmatter.neighbors import covariance, effdim, oas

class CovarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.array([[1, 2], [3, 3], [4, 6]])
        cls.expected_cov = np.array([[2.33333333, 2.83333333],
                                     [2.83333333, 4.33333333]])
        cls.expected_cov_periodic = np.array([[1.12597216, 0.45645371],
                                              [0.45645371, 0.82318948]])
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
        cls.cov = np.array([[1, 1, 0],
                            [1, 1, 0],
                            [0, 0, 1]])
        cls.expected_effdim = 1.8898815748423097

    def test_effdim(self):
        self.assertTrue(np.allclose(effdim(self.cov), self.expected_effdim))

class OASTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cov = np.array([[0.5, 1.0], [0.7, 0.4]])
        cls.n = 10
        cls.D = 2
        cls.expected_oas = np.array([[0.48903924, 0.78078484],
                                     [0.54654939, 0.41096076]])

    def test_oas(self):
        self.assertTrue(np.allclose(oas(self.cov, self.n, self.D), self.expected_oas))
