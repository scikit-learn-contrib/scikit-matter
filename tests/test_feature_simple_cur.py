import unittest

import numpy as np
from sklearn import exceptions

# from sklearn.datasets import load_boston as load
from skcosmo.datasets import load_csd_1000r as load
from skcosmo.feature_selection import CUR


class TestCUR(unittest.TestCase):
    def setUp(self):
        self.X, _ = load(return_X_y=True)

    def test_bad_transform(self):
        selector = CUR(n_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        ref_selector = CUR(n_to_select=self.X.shape[-1] - 3).fit(X=self.X)
        ref_idx = ref_selector.selected_idx_

        selector = CUR(n_to_select=1)
        selector.fit(self.X)

        for i in range(self.X.shape[-1] - 3):
            selector.n_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], ref_idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        C = self.X.T @ self.X
        _, UC = np.linalg.eigh(C)
        ref_idx = np.argsort(-(UC[:, -1] ** 2.0))[:-1]

        selector = CUR(n_to_select=self.X.shape[-1] - 1, iterative=False)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_idx_, ref_idx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
