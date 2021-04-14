import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from skcosmo.sample_selection import (
    FPS,
    VoronoiFPS,
)
from test_sample_simple_fps import TestFPS


class TestVoronoiFPS(TestFPS):
    def setUp(self):
        super().setUp()

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new number of
        features and `warm_start`
        """

        selector = VoronoiFPS(n_to_select=1, initialize=self.idx[0])
        selector.fit(self.X)

        for i in range(2, len(self.idx)):
            selector.n_to_select = i
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i - 1], self.idx[i - 1])

    def test_initialize(self):
        """
        This test checks that the model can be initialized in all applicable manners
        and throws an error otherwise
        """

        for initialize in [self.idx[0], "random"]:
            with self.subTest(initialize=initialize):
                selector = VoronoiFPS(n_to_select=1, initialize=initialize)
                selector.fit(self.X)

        with self.assertRaises(ValueError) as cm:
            selector = VoronoiFPS(n_to_select=1, initialize="bad")
            selector.fit(self.X)
            self.assertEquals(
                str(cm.message), "Invalid value of the initialize parameter"
            )

    def test_switching_point(self):
        """
        This test check work of the switching point calculator into the
        _init_greedy_search function
        """
        selector = VoronoiFPS(n_to_select=1)
        selector.fit(self.X)
        self.assertTrue(1 > selector.full_fraction)

        selector = VoronoiFPS(n_to_select=1, full_fraction=0.5)
        selector.fit(self.X)
        self.assertEqual(selector.full_fraction, 0.5)

        with self.subTest(name="bad_ntrial"):
            with self.assertRaises(ValueError) as cm:
                selector = VoronoiFPS(n_to_select=1, n_trial_calculation=0)
                selector.fit(self.X)
                self.assertEquals(
                    str(cm.message),
                    "Number of trial calculation should be more or equal to 1",
                )

        with self.subTest(name="float_ntrial"):
            with self.assertRaises(TypeError) as cm:
                selector = VoronoiFPS(n_to_select=1, n_trial_calculation=0.3)
                selector.fit(self.X)
                self.assertEquals(
                    str(cm.message), "Number of trial calculation should be integer"
                )

        with self.subTest(name="large_ff"):
            with self.assertRaises(ValueError) as cm:
                selector = VoronoiFPS(n_to_select=1, full_fraction=1.1)
                selector.fit(self.X)
                self.assertEquals(
                    str(cm.message),
                    f"Switching point should be real and more than 0 and less than 1 received {selector.full_fraction}",
                )

        with self.subTest(name="string_ff"):
            with self.assertRaises(ValueError) as cm:
                selector = VoronoiFPS(n_to_select=1, full_fraction="STRING")
                selector.fit(self.X)
                self.assertEquals(
                    str(cm.message),
                    f"Switching point should be real and more than 0 and less than 1 received {selector.full_fraction}",
                )

    def test_get_distances(self):
        """
        This test checks that the haussdorf distances are returnable after fitting
        """
        selector = VoronoiFPS(n_to_select=1)
        selector.fit(self.X)
        _ = selector.get_select_distance()

        with self.assertRaises(NotFittedError):
            selector = VoronoiFPS(n_to_select=1)
            _ = selector.get_select_distance()

    def test_comparison(self):
        """
        This test checks that the voronoi FPS strictly computes less distances
        than its normal FPS counterpart.
        """
        vselector = VoronoiFPS(n_to_select=self.X.shape[-1] - 1)
        vselector.fit(self.X)

        selector = FPS(n_to_select=self.X.shape[-1] - 1)
        selector.fit(self.X)

        self.assertTrue(np.allclose(vselector.selected_idx_, selector.selected_idx_))

    def test_nothing_updated_points(self):
        """
        This test checks that in the case where we have no points to update,
        the code still works fine
        """
        X = np.array([[1, 1], [4, 4], [10, 10], [100, 100]])
        selector = VoronoiFPS(n_to_select=3, initialize=0)
        try:
            selector.fit(X)
            f = 1
        except Exception:
            f = 0
        self.assertEqual(f, 1)

        # print(selector.vlocation_of_idx )
        # self.assertEqual(
        #     len(np.where(selector.vlocation_of_idx == selector.n_selected_)[0]), 1
        # )

    def test_d2q(self):

        selector = VoronoiFPS(n_to_select=3)
        selector.fit(self.X)

        active_points = np.where(
            selector.sel_d2q_[selector.vlocation_of_idx] < selector.haussdorf_
        )[0]

        ap = selector._get_active(self.X, selector.selected_idx_[-1])

        self.assertTrue(
            np.allclose(
                active_points,
                ap,
            )
        )

        selector = VoronoiFPS(n_to_select=1)

        ap = selector._get_active(self.X, 0)

        self.assertTrue(
            np.allclose(
                np.arange(self.X.shape[0]),
                ap,
            )
        )

    def test_calc_disc(self):
        selector = VoronoiFPS(n_to_select=3, initialize=0)
        selector.fit(self.X)

        self.assertTrue(
            np.allclose(
                selector.haussdorf_,
                selector._calculate_distances(self.X, selector.selected_idx_[-1]),
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
