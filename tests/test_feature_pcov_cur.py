import unittest

import numpy as np
from sklearn.datasets import load_boston

from skcosmo.feature_selection import PCovCUR


class TestPCovCUR(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)
        self.idx = [9, 11, 6, 1, 12, 0, 5, 2, 8, 10, 7, 3, 4]

    def test_known(self):
        """
        This test checks that the model returns a known set of indices
        """

        selector = PCovCUR(n_to_select=12)
        selector.fit(self.X, self.y)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx[:-1]))

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        selector = PCovCUR(n_to_select=1)
        selector.fit(self.X, self.y)

        for i in range(len(self.idx) - 2):
            selector.n_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], self.idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        self.idx = [9, 11, 6, 10, 12, 2, 8, 1, 5, 0, 7, 4, 3]
        selector = PCovCUR(n_to_select=12, iterative=False)
        selector.fit(self.X, self.y)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx[:-1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
