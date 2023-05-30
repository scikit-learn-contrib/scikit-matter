import unittest

import numpy as np
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.exceptions import NotFittedError

from skmatter._selection import GreedySelector


class GreedyTester(GreedySelector):
    def __init__(
        self, n_to_select=None, score_threshold=None, selection_type="feature", **kwargs
    ):
        super().__init__(
            selection_type=selection_type,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            **kwargs,
        )

    def score(self, X, y=None):
        scores = np.linalg.norm(X, axis=0)
        scores[self.selected_idx_] = 0.0
        return scores


class TestGreedy(unittest.TestCase):
    def setUp(self):
        self.X, _ = get_dataset(return_X_y=True)

    def test_bad_type(self):
        with self.assertRaises(
            ValueError, msg="Only feature and sample selection supported."
        ):
            _ = GreedyTester(selection_type="bad").fit(self.X)

    def test_score_threshold(self):
        selector = GreedyTester(score_threshold=200, n_to_select=7)
        with self.assertWarns(
            Warning, msg="Score threshold of 200 reached. Terminating search at 6 / 7."
        ):
            selector.fit(self.X)

    def test_score_threshold_and_full(self):
        with self.assertRaises(ValueError) as cm:
            _ = GreedyTester(score_threshold=20, full=True, n_to_select=12).fit(self.X)
        self.assertEqual(
            str(cm.exception),
            "You cannot specify both `score_threshold` and `full=True`.",
        )

    def test_bad_warm_start(self):
        selector = GreedyTester()
        with self.assertRaises(ValueError) as cm:
            selector.fit(self.X, warm_start=True)
        self.assertTrue(
            str(cm.exception),
            "Cannot fit with warm_start=True without having been previously "
            "initialized",
        )

    def test_bad_y(self):
        self.X, self.Y = get_dataset(return_X_y=True)
        Y = self.Y[:2]
        print(self.X.shape, Y.shape)
        selector = GreedyTester(n_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=Y)

    def test_bad_transform(self):
        selector = GreedyTester(n_to_select=2)
        selector.fit(self.X)
        with self.assertRaises(ValueError) as cm:
            _ = selector.transform(self.X[:, :3])
        self.assertEqual(
            str(cm.exception),
            "X has a different shape than during fitting. Reshape your data.",
        )

    def test_no_nfeatures(self):
        selector = GreedyTester()
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), self.X.shape[1] // 2)

    def test_decimal_nfeatures(self):
        selector = GreedyTester(n_to_select=0.2)
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), int(self.X.shape[1] * 0.2))

    def test_bad_nfeatures(self):
        for nf in [1.2, "1", 20]:
            with self.subTest(n_features=nf):
                selector = GreedyTester(n_to_select=nf)
                with self.assertRaises(ValueError) as cm:
                    selector.fit(self.X)
                self.assertEqual(
                    str(cm.exception),
                    (
                        "n_to_select must be either None, an integer in "
                        "[1, n_features] representing the absolute number "
                        "of features, or a float in (0, 1] representing a "
                        f"percentage of features to select. Got {nf} "
                        f"features and an input with {self.X.shape[1]} feature."
                    ),
                )

    def test_not_fitted(self):
        with self.assertRaises(NotFittedError):
            selector = GreedyTester()
            _ = selector._get_support_mask()

    def test_fitted(self):
        selector = GreedyTester()
        selector.fit(self.X)
        _ = selector._get_support_mask()

        Xr = selector.transform(self.X)
        self.assertEqual(Xr.shape[1], self.X.shape[1] // 2)

    def test_size_input(self):
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        selector_sample = GreedyTester(selection_type="sample")
        selector_feature = GreedyTester(selection_type="feature")
        with self.assertRaises(ValueError) as cm:
            selector_feature.fit(X)
        self.assertEqual(
            str(cm.exception),
            f"Found array with 1 feature(s) (shape={X.shape})"
            " while a minimum of 2 is required.",
        )

        X = X.reshape(1, -1)

        with self.assertRaises(ValueError) as cm:
            selector_sample.fit(X)
        self.assertEqual(
            str(cm.exception),
            f"Found array with 1 sample(s) (shape={X.shape}) while a minimum of 2 is "
            "required.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
