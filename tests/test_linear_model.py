import unittest
from parameterized import parameterized

import numpy as np
from scipy.stats import ortho_group
from sklearn.datasets import load_iris

from skcosmo.linear_model import OrthogonalRegression


class BaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        features = load_iris().data
        cls.features_small = features[:, [0, 1]]
        cls.features_large = features[:, [0, 1, 0, 1]]
        cls.eps = 1e-9
        np.random.seed(0x5F3759DF)
        cls.features_rotated_small = cls.features_small.dot(
            ortho_group.rvs(cls.features_small.shape[1])
        )

    def test_orthogonal_regression_small_to_rotated_small(self):
        # tests if OrthogonalRegression can predict rotated small features using small features with use_orthogonal_projector False
        err = np.linalg.norm(
            self.features_rotated_small
            - OrthogonalRegression(use_orthogonal_projector=False)
            .fit(self.features_small, self.features_rotated_small)
            .predict(self.features_small)
        )
        self.assertTrue(
            abs(err) < self.eps, f"error {err} surpasses threshold for zero {self.eps}"
        )

    def test_orthogonal_regression_large_to_small(self):
        # tests if prediction is padded to larger feature size
        n_features = (
            OrthogonalRegression(use_orthogonal_projector=False)
            .fit(self.features_large, self.features_small)
            .predict(self.features_large)
            .shape[1]
        )
        self.assertTrue(
            n_features == self.features_large.shape[1],
            f"n_features {n_features} does not match larger feature size {self.features_large.shape[1]}",
        )

    def test_orthogonal_regression_use_orthogonal_projector_small_to_rotated_small(
        self,
    ):
        # tests if OrthogonalRegression can predict rotated small features using small features with use_orthogonal_projector True
        err = np.linalg.norm(
            self.features_rotated_small
            - OrthogonalRegression(use_orthogonal_projector=True)
            .fit(self.features_small, self.features_rotated_small)
            .predict(self.features_small)
        )
        self.assertTrue(
            abs(err) < self.eps, f"error {err} surpasses threshold for zero {self.eps}"
        )

    def test_orthogonal_regression_use_orthogonal_projector_small_to_large(self):
        # tests if prediction is projected to prediction feature space
        n_features = (
            OrthogonalRegression(use_orthogonal_projector=True)
            .fit(self.features_small, self.features_large)
            .predict(self.features_small)
            .shape[1]
        )
        self.assertTrue(
            n_features == self.features_large.shape[1],
            f"n_features {n_features} does not match projection feature size {self.features_large.shape[1]}",
        )

    def test_orthogonal_regression_use_orthogonal_projector_large_to_small(self):
        # tests if prediction is projected to prediction feature space
        n_features = (
            OrthogonalRegression(use_orthogonal_projector=True)
            .fit(self.features_large, self.features_small)
            .predict(self.features_large)
            .shape[1]
        )
        self.assertTrue(
            n_features == self.features_small.shape[1],
            f"n_features {n_features} does not match projection feature size {self.features_small.shape[1]}",
        )


if __name__ == "__main__":
    unittest.main()
