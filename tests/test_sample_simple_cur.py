import unittest
import numpy as np

from sklearn.datasets import load_boston
from sklearn import exceptions

from skcosmo.sample_selection import CUR


class TestCUR(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)
        self.idx = [488, 450, 183, 199, 380, 483, 214, 126, 374, 8]

    def test_bad_y(self):
        selector = CUR(n_samples_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=self.y[:2])

    def test_good_y(self):
        selector = CUR(n_samples_to_select=2)
        selector.fit(X=self.X, y=self.y)
        self.assertTrue(selector.y_current is not None)

    def test_no_y(self):
        selector = CUR(n_samples_to_select=2)
        selector.fit(X=self.X, y=None)
        self.assertTrue(selector.y_current is None)

    def test_bad_transform(self):
        selector = CUR(n_samples_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        selector = CUR(n_samples_to_select=1)
        selector.fit(self.X)

        for i in range(len(self.idx) - 2):
            selector.n_samples_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], self.idx[i])

    def test_restart_with_y(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        selector = CUR(n_samples_to_select=1)
        selector.fit(self.X, self.y)

        for i in range(len(self.idx) - 2):
            selector.n_samples_to_select += 1
            selector.fit(self.X, self.y, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], self.idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        self.idx = [488, 492, 491, 489, 398, 374, 373, 386, 400, 383]
        selector = CUR(n_samples_to_select=len(self.idx), iterative=False)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_idx_, self.idx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
