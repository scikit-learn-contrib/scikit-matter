import unittest

import numpy as np

from skmatter.clustering import QuickShift


class QuickShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = np.array(
            [
                [-1.72779275, -1.32763554],
                [-4.44991964, -2.13474901],
                [0.54817734, -2.43319467],
                [3.19881307, -0.49547222],
                [-1.1335991, 2.33478428],
                [0.55437388, 0.18745963],
            ]
        )
        cls.cuts = np.array(
            [6.99485011, 8.80292681, 7.68486852, 9.5115009, 8.07736919, 6.22057056]
        )
        cls.weights = np.array(
            [
                -3.94008092,
                -12.68095664,
                -7.07512499,
                -9.03064023,
                -8.26529849,
                -2.61132267,
            ]
        )
        cls.labels_ = np.array([0, 0, 0, 5, 5, 5])
        cls.cluster_centers_idx_ = np.array([0, 5])
        cls.cell = [3, 3]

    def test_fit(self):
        model = QuickShift(self.cuts)
        model.fit(self.points, samples_weight=self.weights)
        self.assertTrue(np.all(model.labels_ == self.labels_))
        self.assertTrue(np.all(model.cluster_centers_idx_ == self.cluster_centers_idx_))

    def test_dimension_check(self):
        model = QuickShift(self.cuts, metric_params={"cell": self.cell})
        self.assertRaises(ValueError, model.fit, np.array([[2]]))
