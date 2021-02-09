import unittest
import numpy as np

from sklearn.datasets import load_boston
from sklearn import exceptions

from skcosmo.feature_selection import SimpleFPS


class TestSimpleFPS(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)
        self.idx = [9, 3, 11, 6, 1, 10, 8, 0, 12, 2, 5, 7, 4]

    def test_bad_y(self):
        self.X, self.Y = load_boston(return_X_y=True)
        selector = SimpleFPS(n_features_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=self.Y[:2])

    def test_bad_transform(self):
        selector = SimpleFPS(n_features_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        norms = (self.X ** 2).sum(axis=0)
        hd = norms + norms[self.idx[0]] - 2 * (self.X.T @ self.X[:, self.idx[0]])

        for i in range(2, len(self.idx)):
            selector = SimpleFPS(n_features_to_select=i)
            selector.fit(self.X, initial=self.idx[: i - 1], haussdorfs=hd)
            hd = selector.haussdorf_
            self.assertEqual(selector.selected_[-1], self.idx[i - 1])

        with self.subTest(type="bad haussdorf"):
            with self.assertRaises(ValueError) as cm:
                selector = SimpleFPS(n_features_to_select=i)
                selector.fit(self.X, initial=self.idx[:2], haussdorfs=hd[:3])
                self.assertEqual(
                    str(cm.message),
                    "The number of pre-computed haussdorf distances"
                    "does not match the number of features.",
                )

    def test_selected_distance(self):
        """
        This test checks that the model can be restarted with a new instance
        """

        hd = np.zeros(len(self.idx) - 1)
        for i in range(2, len(self.idx)):
            selector = SimpleFPS(n_features_to_select=i)
            selector.fit(self.X, initial=self.idx[: i - 1])
            hd[i - 1] = selector.haussdorf_[selector.selected_[i - 1]]

        selector = SimpleFPS(n_features_to_select=len(self.idx) - 1)
        selector.fit(self.X, initial=[self.idx[0]])

        self.assertTrue(np.allclose(selector.get_select_distance(self.X), hd))

    def test_supplied_indices(self):
        """
        This test checks FPS will match pre-defined indices.
        """

        selector = SimpleFPS(n_features_to_select=len(self.idx) - 1)
        selector.fit(self.X, initial=self.idx)
        for i in range(len(self.idx) - 1):
            with self.subTest(i=i, idx=self.idx[i]):
                self.assertEqual(selector.selected_[i], self.idx[i])

    def test_partially_supplied_indices(self):
        """
        This test checks FPS will accept pre-defined indices.
        """

        selector = SimpleFPS(n_features_to_select=len(self.idx) - 1)
        selector.fit(self.X, initial=self.idx[:3])
        for i in range(len(self.idx) - 1):
            with self.subTest(i=i, idx=self.idx[i]):
                self.assertEqual(selector.selected_[i], self.idx[i])

    def test_no_nfeatures(self):
        selector = SimpleFPS()
        selector.fit(self.X, initial=self.idx[0])
        self.assertEqual(len(selector.selected_), self.X.shape[1] // 2)

    def test_decimal_nfeatures(self):
        selector = SimpleFPS(n_features_to_select=0.2)
        selector.fit(self.X, initial=self.idx[0])
        self.assertEqual(len(selector.selected_), int(self.X.shape[1] * 0.2))

    def test_bad_nfeatures(self):
        for nf in [1.2, "1", 20]:
            with self.subTest(n_features=nf):
                selector = SimpleFPS(n_features_to_select=nf)
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
