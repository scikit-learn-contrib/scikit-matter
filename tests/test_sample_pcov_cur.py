import unittest

import numpy as np
from sklearn.datasets import load_boston

from skcosmo.sample_selection import PCovCUR

EPSILON = 1e-6


class TestPCovCUR(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)
        self.idx = [488, 283, 183, 380, 41, 438, 368, 374, 123, 353]

    def test_known(self):
        """
        This test checks that the model returns a known set of indices
        """

        selector = PCovCUR(n_to_select=10, mixing=0.5, iterative=True)
        selector.fit(self.X, self.y)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx))

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        selector = PCovCUR(n_to_select=1)
        selector.fit(self.X, self.y)

        for i in range(len(self.idx) - 2):
            selector.n_to_select += 1
            selector.fit(self.X, self.y, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], self.idx[i])

            self.assertLessEqual(
                np.linalg.norm(selector.X_current_[self.idx[i]]), EPSILON
            )

            for j in range(self.X.shape[0]):
                self.assertLessEqual(
                    np.dot(selector.X_current_[self.idx[i]], selector.X_current_[j]),
                    EPSILON,
                )

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        self.idx = [488, 492, 491, 374, 398, 373, 386, 400, 383, 382]
        selector = PCovCUR(n_to_select=10, iterative=False)
        selector.fit(self.X, self.y)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx))

    def test_multiple_k(self):
        """
        This test checks that the model can be run with multiple k's
        """

        for k in np.logspace(0, np.log10(self.X.shape[0]), 4, dtype=int):
            selector = PCovCUR(n_to_select=10, k=k)
            selector.fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
