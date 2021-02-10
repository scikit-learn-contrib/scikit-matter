import unittest
import numpy as np

from sklearn.datasets import load_boston
from sklearn import exceptions

from skcosmo.feature_selection import SimpleCUR


class TestSimpleCUR(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)
        self.idx = [9, 11, 6, 1, 0, 12, 2, 8, 10, 7, 5, 3, 4]

    def test_bad_y(self):
        self.X, self.Y = load_boston(return_X_y=True)
        selector = SimpleCUR(n_features_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=self.Y[:2])

    def test_bad_transform(self):
        selector = SimpleCUR(n_features_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        X_current = self.X.copy()

        for i in range(len(self.idx) - 1):
            selector = SimpleCUR(n_features_to_select=i + 1)
            selector.fit(X_current, initial=self.idx[:i])
            X_current = selector.X_current.copy()
            self.assertEqual(selector.selected_[-1], self.idx[i])

    def test_non_it(self):
        """
        This test checks that the model can be run non-iteratively
        """
        self.idx = [9, 11, 6, 10, 12, 2, 8, 1, 5, 0, 7, 4, 3]
        selector = SimpleCUR(n_features_to_select=12, iterative=False)
        selector.fit(self.X)

        self.assertTrue(np.allclose(selector.selected_, self.idx[:-1]))

    def test_supplied_indices(self):
        """
        This test checks FPS will match pre-defined indices.
        """

        selector = SimpleCUR(n_features_to_select=len(self.idx) - 1)
        selector.fit(self.X, initial=self.idx)
        for i in range(len(self.idx) - 1):
            with self.subTest(i=i, idx=self.idx[i]):
                self.assertEqual(selector.selected_[i], self.idx[i])

    def test_no_nfeatures(self):
        selector = SimpleCUR()
        selector.fit(self.X, initial=self.idx[0])
        self.assertEqual(len(selector.selected_), self.X.shape[1] // 2)

    def test_decimal_nfeatures(self):
        selector = SimpleCUR(n_features_to_select=0.2)
        selector.fit(self.X, initial=self.idx[0])
        self.assertEqual(len(selector.selected_), int(self.X.shape[1] * 0.2))

    def test_bad_nfeatures(self):
        for nf in [1.2, "1", 20]:
            with self.subTest(n_features=nf):
                selector = SimpleCUR(n_features_to_select=nf)
                with self.assertRaises(ValueError) as cm:
                    selector.fit(self.X, initial=self.idx[0])
                    self.assertEqual(
                        str(cm.message),
                        (
                            "n_features_to_select must be either None, an "
                            "integer in [1, n_features - 1] "
                            "representing the absolute "
                            "number of features, or a float in (0, 1] "
                            "representing a percentage of features to "
                            f"select. Got {nf}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
