import unittest
import numpy as np

from sklearn.exceptions import NotFittedError
from skcosmo.datasets import load_csd_1000r

from skcosmo.feature_selection.voronoi_fps import VoronoiFPS
from skcosmo.feature_selection import FPS


class TestVoronoiFPS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_csd_1000r(return_X_y=True)
        # re-order features, as they were already FPS-ed
        norms = np.linalg.norm(self.X, axis=0)
        self.X = self.X[:, np.argsort(norms)]
        self.idx = [
            0,
            99,
            98,
            97,
            96,
            95,
            93,
            94,
            92,
            91,
            90,
            88,
            86,
            85,
            79,
            78,
            87,
            83,
            81,
            77,
            82,
            71,
            84,
            67,
            74,
            64,
            65,
            62,
            76,
            70,
            68,
            69,
            56,
            63,
            46,
            55,
            58,
            53,
            72,
            73,
            50,
            60,
            49,
            66,
            47,
            44,
            45,
            57,
            48,
            43,
            31,
            39,
            61,
            38,
            26,
            89,
            25,
            27,
            36,
            41,
            12,
            33,
            11,
            10,
            51,
            22,
            14,
            40,
            75,
            42,
            18,
            17,
            35,
            54,
            21,
            59,
            80,
            15,
            20,
            28,
            9,
            32,
            37,
            5,
            3,
            13,
            23,
            34,
            30,
            16,
            24,
            8,
            4,
            52,
            29,
            7,
            6,
            2,
            19,
        ]

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new number of
        features and `warm_start`
        """

        selector = VoronoiFPS(n_features_to_select=1, initialize=self.idx[0])
        selector.fit(self.X)

        for i in range(2, len(self.idx)):
            selector.n_features_to_select = i
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i - 1], self.idx[i - 1])

    def test_initialize(self):
        """
        This test checks that the model can be initialized in all applicable manners
        and throws an error otherwise
        """

        for initialize in [self.idx[0], "random"]:
            with self.subTest(initialize=initialize):
                selector = VoronoiFPS(n_features_to_select=1, initialize=initialize)
                selector.fit(self.X)

        with self.assertRaises(ValueError) as cm:
            selector = VoronoiFPS(n_features_to_select=1, initialize="bad")
            selector.fit(self.X)
            self.assertEquals(
                str(cm.message), "Invalid value of the initialize parameter"
            )

    def test_get_distances(self):
        """
        This test checks that the haussdorf distances are returnable after fitting
        """
        selector = VoronoiFPS(n_features_to_select=1)
        selector.fit(self.X)
        _ = selector.get_select_distance()

        with self.assertRaises(NotFittedError):
            selector = VoronoiFPS(n_features_to_select=1)
            _ = selector.get_select_distance()

    def test_comparison(self):
        """
        This test checks that the voronoi FPS strictly computes less distances
        than its normal FPS counterpart.
        """
        vselector = VoronoiFPS(n_features_to_select=self.X.shape[-1] - 1)
        vselector.fit(self.X)

        selector = FPS(n_features_to_select=self.X.shape[-1] - 1)
        selector.fit(self.X)

        self.assertTrue(np.allclose(vselector.selected_idx_, selector.selected_idx_))


if __name__ == "__main__":
    unittest.main(verbosity=2)
