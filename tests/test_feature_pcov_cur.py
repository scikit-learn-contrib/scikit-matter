import unittest

import numpy as np
from sklearn.datasets import load_diabetes as get_dataset

from skcosmo.feature_selection import PCovCUR


class TestPCovCUR(unittest.TestCase):
    def setUp(self):
        self.X, self.y = get_dataset(return_X_y=True)
        self.idx = [2, 8, 3, 4, 1, 7, 5, 9, 6]

    def test_known(self):
        """
        This test checks that the model returns a known set of indices
        """

        selector = PCovCUR(n_to_select=9)
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
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], self.idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        self.idx = [2, 8, 3, 6, 7, 9, 1, 0, 5]
        selector = PCovCUR(n_to_select=9, recompute_every=0)
        selector.fit(self.X, self.y)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
