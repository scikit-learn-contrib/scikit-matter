import unittest

import numpy as np
from sklearn import exceptions

from skcosmo.datasets import load_csd_1000r as load
from skcosmo.sample_selection import CUR


class TestCUR(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load(return_X_y=True)
        self.n_select = min(self.X.shape) // 2

    def test_bad_transform(self):
        selector = CUR(n_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        ref_selector = CUR(n_to_select=self.n_select)
        ref_idx = ref_selector.fit(self.X).selected_idx_

        selector = CUR(n_to_select=1)
        selector.fit(self.X)

        for i in range(len(ref_idx) - 2):
            selector.n_to_select += 1
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i], ref_idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """

        K = self.X @ self.X.T
        _, UK = np.linalg.eigh(K)
        ref_idx = np.argsort(-(UK[:, -1] ** 2.0))[: self.n_select]

        selector = CUR(n_to_select=len(ref_idx), iterative=False)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_idx_, ref_idx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
