import unittest

import numpy as np
from parameterized import parameterized
from sklearn.datasets import load_iris
from sklearn.utils import (
    check_random_state,
    extmath,
)

from skmatter.linear_model import (
    OrthogonalRegression,
    RidgeRegression2FoldCV,
)


class BaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features_all = load_iris().data
        cls.features_small = cls.features_all[:, [0, 1]]
        cls.features_large = cls.features_all[:, [0, 1, 0, 1]]
        cls.eps = 1e-9
        random_state = 0
        random_state = check_random_state(random_state)
        random_orthonormal_mat = extmath.randomized_range_finder(
            np.eye(cls.features_small.shape[1]),
            size=cls.features_small.shape[1],
            n_iter=10,
            random_state=random_state,
        )
        cls.features_rotated_small = cls.features_small @ random_orthonormal_mat

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


class RidgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features_all = load_iris().data
        cls.features_small = cls.features_all[:, [0, 1]]
        cls.features_large = cls.features_all[:, [0, 1, 0, 1]]
        cls.eps = 5e-8
        np.random.RandomState(0).seed(0x5F3759DF)
        cls.alphas = [1e-9, 1e-3, 1e-1, 0.5]
        cls.ridge_regressions = []

    def test_ridge_regression_2fold_regularization_method_raise_error(self):
        # tests if wrong regularization_method in RidgeRegression2FoldCV raises error
        with self.assertRaises(ValueError):
            RidgeRegression2FoldCV(
                regularization_method="dummy",
            ).fit(self.features_small, self.features_small)

    def test_ridge_regression_2fold_alpha_type_raise_error(self):
        # tests if wrong alpha type in RidgeRegression2FoldCV raises error
        with self.assertRaises(ValueError):
            RidgeRegression2FoldCV(
                alpha_type="dummy",
            ).fit(self.features_small, self.features_small)

    def test_ridge_regression_2fold_relative_alpha_type_raise_error(self):
        # tests if an error is raised if alpha not in [0,1)
        with self.assertRaises(ValueError):
            RidgeRegression2FoldCV(alphas=[1], alpha_type="relative").fit(
                self.features_small, self.features_small
            )

        with self.assertRaises(ValueError):
            RidgeRegression2FoldCV(alphas=[-0.1], alpha_type="relative").fit(
                self.features_small, self.features_small
            )

    ridge_parameters = [
        ["absolute_tikhonov", "absolute", "tikhonov"],
        ["absolute_cutoff", "absolute", "cutoff"],
        ["relative_tikhonov", "relative", "tikhonov"],
        ["relative_cutoff", "relative", "cutoff"],
    ]

    @parameterized.expand(ridge_parameters)
    def test_ridge_regression_2fold_cv_small_to_small(
        self, name, alpha_type, regularization_method
    ):
        # tests if RidgeRegression2FoldCV can predict small features using small features with use_orthogonal_projector False
        err = np.linalg.norm(
            self.features_small
            - RidgeRegression2FoldCV(
                alphas=self.alphas,
                alpha_type=alpha_type,
                regularization_method=regularization_method,
            )
            .fit(self.features_small, self.features_small)
            .predict(self.features_small)
        )
        self.assertTrue(
            abs(err) < self.eps, f"error {err} surpasses threshold for zero {self.eps}"
        )

    @parameterized.expand(ridge_parameters)
    def test_ridge_regression_2fold_cv_small_to_large(
        # tests if RidgeRegression2FoldCV can predict large features using small features with use_orthogonal_projector False
        self,
        name,
        alpha_type,
        regularization_method,
    ):
        err = np.linalg.norm(
            self.features_large
            - RidgeRegression2FoldCV(
                alphas=self.alphas,
                alpha_type=alpha_type,
                regularization_method=regularization_method,
            )
            .fit(self.features_small, self.features_large)
            .predict(self.features_small)
        )
        self.assertTrue(
            abs(err) < self.eps,
            f"error {err} surpasses threshold for zero {self.eps}",
        )

    @parameterized.expand(ridge_parameters)
    def test_ridge_regression_2fold_regularization(
        self, name, alpha_type, regularization_method
    ):
        # tests if the regularization in the CV split of
        # RidgeRegression2FoldCV does effect the results

        # regularization parameters are chosen to match the singular values o
        # the features, thus each regularization parameter affects the minimized
        # weight matrix and thus the error
        _, singular_values, _ = np.linalg.svd(self.features_all)
        if alpha_type == "absolute":
            alphas = singular_values[1:][::-1]
        if alpha_type == "relative":
            alphas = singular_values[1:][::-1] / singular_values[0]

        # tests if RidgeRegression2FoldCV does do regularization correct
        ridge = RidgeRegression2FoldCV(
            alphas=alphas,
            alpha_type=alpha_type,
            regularization_method=regularization_method,
            scoring="neg_root_mean_squared_error",
        ).fit(self.features_all, self.features_all)
        twofold_rmse = -np.array(ridge.cv_values_)

        # since the data can be perfectly reconstructed,
        # larger regularization parameters (alphas) should result in
        # larger errors
        error_grad = twofold_rmse[1:] - twofold_rmse[:-1]
        self.assertTrue(
            np.all(error_grad > self.eps),
            "error does not strictly increase with larger regularization\n"
            f"\ttwofold RMSE: {twofold_rmse}\n"
            f"\tregularization parameters: {ridge.alphas}",
        )


if __name__ == "__main__":
    unittest.main()
