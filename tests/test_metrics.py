import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state, extmath

from skmatter.datasets import load_degenerate_CH4_manifold
from skmatter.metrics import (
    componentwise_prediction_rigidity,
    global_reconstruction_distortion,
    global_reconstruction_error,
    local_prediction_rigidity,
    local_reconstruction_error,
    pairwise_euclidean_distances,
    pairwise_mahalanobis_distances,
    pointwise_local_reconstruction_error,
)


class PredictionRigidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        soap_features = load_degenerate_CH4_manifold().data["SOAP_power_spectrum"]
        soap_features = soap_features[:11]
        # each structure in CH4 has 5 environmental feature, because there are 5 atoms
        # per structure and each atom is one environment
        cls.features = [
            soap_features[i * 5 : (i + 1) * 5] for i in range(len(soap_features) // 5)
        ]
        # add a single environment structure to check value
        cls.features = cls.features + [soap_features[-1:]]
        cls.alpha = 1e-8
        bi_features = load_degenerate_CH4_manifold().data["SOAP_bispectrum"]
        bi_features = bi_features[:11]
        comp_features = np.column_stack([soap_features, bi_features])
        cls.comp_features = [
            comp_features[i * 5 : (i + 1) * 5] for i in range(len(comp_features) // 5)
        ]
        cls.comp_dims = np.array([soap_features.shape[1], bi_features.shape[1]])

    def test_local_prediction_rigidity(self):
        LPR, rank_diff = local_prediction_rigidity(
            self.features, self.features, self.alpha
        )
        self.assertTrue(
            LPR[-1] >= 1,
            f"LPR of the single environment structure is incorrectly lower than 1:"
            f"LPR = {LPR[-1]}",
        )
        self.assertTrue(
            rank_diff == 0,
            f"LPR Covariance matrix rank is not full, with a difference of:"
            f"{rank_diff}",
        )

    def test_componentwise_prediction_rigidity(self):
        _CPR, _LCPR, _rank_diff = componentwise_prediction_rigidity(
            self.comp_features, self.comp_features, self.alpha, self.comp_dims
        )


class ReconstructionMeasuresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        features = load_iris().data
        cls.features_small = features[:20, [0, 1]]
        cls.features_large = features[:20, [0, 1, 0, 1]]
        cls.eps = 1e-5
        cls.n_local_points = 15

        random_state = 0
        random_state = check_random_state(random_state)
        random_orthonormal_mat = extmath.randomized_range_finder(
            np.eye(cls.features_small.shape[1]),
            size=cls.features_small.shape[1],
            n_iter=10,
            random_state=random_state,
        )
        cls.features_rotated_small = cls.features_small @ random_orthonormal_mat

    def test_global_reconstruction_error_identity(self):
        gfre_val = global_reconstruction_error(self.features_large, self.features_large)
        self.assertTrue(
            abs(gfre_val) < self.eps,
            f"global_reconstruction_error {gfre_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_global_reconstruction_error_small_to_large(self):
        # tests that the GRE of a small set of features onto a larger set of features
        # returns within a threshold of zero
        gfre_val = global_reconstruction_error(self.features_small, self.features_large)
        self.assertTrue(
            abs(gfre_val) < self.eps,
            f"global_reconstruction_error {gfre_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_global_reconstruction_error_large_to_small(self):
        # tests that the GRE of a large set of features onto a smaller set of features
        # returns within a threshold of zero
        gfre_val = global_reconstruction_error(self.features_large, self.features_small)
        self.assertTrue(
            abs(gfre_val) < self.eps,
            f"global_reconstruction_error {gfre_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_global_reconstruction_distortion_identity(self):
        # tests that the GRD of a set of features onto itself returns within a threshold
        # of zero
        gfrd_val = global_reconstruction_distortion(
            self.features_large, self.features_large
        )
        self.assertTrue(
            abs(gfrd_val) < self.eps,
            f"global_reconstruction_error {gfrd_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_global_reconstruction_distortion_small_to_large(self):
        # tests that the GRD of a small set of features onto a larger set of features
        # returns within a threshold of zero
        # should just run
        global_reconstruction_error(self.features_small, self.features_large)

    def test_global_reconstruction_distortion_large_to_small(self):
        # tests that the GRD of a large set of features onto a smaller set of features
        # returns within a threshold of zero
        # should just run
        global_reconstruction_error(self.features_large, self.features_small)

    def test_global_reconstruction_distortion_small_to_rotated_small(self):
        # tests that the GRD of a small set of features onto a rotation of itself
        # returns within a threshold of zero
        gfrd_val = global_reconstruction_distortion(
            self.features_small, self.features_rotated_small
        )
        self.assertTrue(
            abs(gfrd_val) < self.eps,
            f"global_reconstruction_error {gfrd_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_local_reconstruction_error_identity(self):
        # tests that the local reconstruction error of a set of features onto itself
        # returns within a threshold of zero

        lfre_val = local_reconstruction_error(
            self.features_large, self.features_large, self.n_local_points
        )
        self.assertTrue(
            abs(lfre_val) < self.eps,
            f"local_reconstruction_error {lfre_val} surpasses threshold for zero"
            f" {self.eps}",
        )

    def test_local_reconstruction_error_small_to_large(self):
        # tests that the local reconstruction error of a small set of features onto a
        # larger set of features returns within a threshold of zero

        lfre_val = local_reconstruction_error(
            self.features_small, self.features_large, self.n_local_points
        )
        self.assertTrue(
            abs(lfre_val) < self.eps,
            f"local_reconstruction_error {lfre_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_local_reconstruction_error_large_to_small(self):
        # tests that the local reconstruction error of a larger set of features onto a
        # smaller set of features returns within a threshold of zero

        lfre_val = local_reconstruction_error(
            self.features_large, self.features_small, self.n_local_points
        )
        self.assertTrue(
            abs(lfre_val) < self.eps,
            f"local_reconstruction_error {lfre_val} surpasses threshold for zero "
            f"{self.eps}",
        )

    def test_local_reconstruction_error_train_idx(self):
        # tests that the local reconstruction error works when specifying a manual
        # train idx

        lfre_val = pointwise_local_reconstruction_error(
            self.features_large,
            self.features_large,
            self.n_local_points,
            train_idx=np.arange((len(self.features_large) // 4)),
        )
        test_size = len(self.features_large) - (len(self.features_large) // 4)
        self.assertTrue(
            len(lfre_val) == test_size,
            f"size of pointwise LFRE  {len(lfre_val)} differs from expected test set "
            f"size {test_size}",
        )

    def test_local_reconstruction_error_test_idx(self):
        # tests that the local reconstruction error works when specifying a manual
        # train idx

        lfre_val = pointwise_local_reconstruction_error(
            self.features_large,
            self.features_large,
            self.n_local_points,
            test_idx=np.arange((len(self.features_large) // 4)),
        )
        test_size = len(self.features_large) // 4
        self.assertTrue(
            len(lfre_val) == test_size,
            f"size of pointwise LFRE  {len(lfre_val)} differs from expected test set "
            f"size {test_size}",
        )


class DistanceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = [[1, 2], [3, 4], [5, 6]]
        cls.Y = [[7, 8], [9, 10]]
        cls.covs = np.array([[[1, 0.5], [0.5, 1]], [[1, 0.0], [0.0, 1]]])
        cls.cell = [5, 7]
        cls.distances = np.array(
            [
                [8.48528137, 11.3137085],
                [5.65685425, 8.48528137],
                [2.82842712, 5.65685425],
            ]
        )
        cls.periodic_distances = np.array(
            [
                [1.41421356, 2.23606798],
                [3.16227766, 1.41421356],
                [2.82842712, 3.16227766],
            ]
        )
        cls.mahalanobis_distances = np.array(
            [
                [
                    [10.39230485, 13.85640646],
                    [6.92820323, 10.39230485],
                    [3.46410162, 6.92820323],
                ],
                cls.distances,
            ]
        )

    def test_euclidean_distance(self):
        distances = pairwise_euclidean_distances(self.X, self.Y)
        self.assertTrue(
            np.allclose(distances, self.distances),
            f"Calculated distance does not match expected value"
            f"Calculated: {distances} Expected: {self.distances}",
        )

    def test_periodic_euclidean_distance(self):
        distances = pairwise_euclidean_distances(self.X, self.Y, cell=self.cell)
        self.assertTrue(
            np.allclose(distances, self.periodic_distances),
            f"Calculated distance does not match expected value"
            f"Calculated: {distances} Expected: {self.periodic_distances}",
        )

    def test_mahalanobis_distance(self):
        distances = pairwise_mahalanobis_distances(self.X, self.Y, self.covs)
        self.assertTrue(
            np.allclose(distances, self.mahalanobis_distances),
            f"Calculated distance does not match expected value"
            f"Calculated: {distances} Expected: {self.mahalanobis_distances}",
        )


if __name__ == "__main__":
    unittest.main()
