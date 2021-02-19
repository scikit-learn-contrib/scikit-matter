import unittest
import numpy as np

from sklearn.datasets import load_boston
from sklearn import exceptions

from skcosmo.feature_selection import CUR


class TestCUR(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)
        self.idx = [9, 11, 6, 1, 0, 12, 2, 8, 10, 7, 5, 3, 4]

    def test_bad_y(self):
        self.X, self.Y = load_boston(return_X_y=True)
        selector = CUR(n_features_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=self.Y[:2])

    def test_bad_transform(self):
        selector = CUR(n_features_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        selector = CUR(n_features_to_select=1)
        selector.fit(self.X)

        for i in range(len(self.idx) - 2):
            selector.n_features_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], self.idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        self.idx = [9, 11, 6, 10, 12, 2, 8, 1, 5, 0, 7, 4, 3]
        selector = CUR(n_features_to_select=12, iterative=False)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx[:-1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
